#encoding=utf-8
import argparse
import os
from pickle import FALSE, TRUE
import sys
import time
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils import clip_grad_norm_

import torch.distributed as dist
from torch.nn import SyncBatchNorm
import torch.multiprocessing as mp

from utils.logger import get_logger_simple
from utils.convert import tensor_to_cuda
import utils.exp_utils as exp_utils
import utils.evaluator_utils as eval_utils
from utils.evaluator_utils import AverageMeter, ProgressMeter
from utils import lr_utils

import warnings

warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3',
                        help='assign which gpu to use.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='train batch size.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of gpus for distributed training')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--nr', type=int, default=0,
                        help='node rank.')
    parser.add_argument('--rank', type=int, default=0,
                        help='node rank.')
    parser.add_argument('--port', type=str, default='24455',
                        help='process rank.')
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--mode', type=str, default='train',
                        help='one out of : train / infer / train_infer')
    parser.add_argument('--workers', type=int, default=1,
                        help='numbers of workers.')
    parser.add_argument('--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    parser.add_argument('--add_name', type=str, default='',
                        help='additiaonal name of out csv.')
    parser.add_argument('--use_ema_checkpoint', action='store_true', default=False,
                        help='use average checkpoints to predict.')
    parser.add_argument('--use_agv_checkpoint', action='store_true', default=False,
                        help='use average checkpoints to predict.')
    parser.add_argument('--use_multi_checkpoint', action='store_true', default=False,
                        help='use multi checkpoints to predict.')
    parser.add_argument('--use_latest_checkpoint', action='store_true', default=False,
                        help='use latest checkpoints to predict.')
    parser.add_argument('--only_val', action='store_true', default=False,
                        help='only validation.')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resuming to checkpoint and continue training.')
    parser.add_argument('--load_checkpoint_only', action='store_true', default=False,
                        help='resuming to checkpoint and continue training.')
    parser.add_argument('--use_stored_settings', action='store_true', default=False,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--exp_source', type=str, default= 'experiments/',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    parser.add_argument('--stored_source', type=str, default='./output/nodule_generation/sources',
                        help='specifies, from which source experiment to load configs, data_loader and model.')

    parser.add_argument('--fix_lr', action='store_true', default=False,
                    help='Fix learning rate for the predictor')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.ngpus = torch.cuda.device_count()

    cf = exp_utils.prep_exp_ddp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=True)
    
    if args.world_size == -1:
        args.world_size = args.ngpus * args.nodes

    if args.distributed:
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
        mp.spawn(main_worker, nprocs=args.ngpus, args=(args, cf))
    else:
        # Simply call main_worker function
        args.gpu = 0
        main_worker(args.gpu, args, cf)


def main_worker(local_rank, args, cf):
    # import model and dataset
    model = exp_utils.import_module('model', cf.model_path)
    if args.use_stored_settings:
        sys.path.append(cf.save_sources)
        import dataset
    else:
        sys.path.append(args.exp_source)
        import dataset

    logger = get_logger_simple(cf.log_dir)
    logger.info("performing training with model {}, use gpu {}".format(cf.model, local_rank))

    if args.distributed:
        # global rank
        args.rank = args.nr * args.ngpus + local_rank
        setup(args.rank, args.world_size, args.port)

    # infer learning rate before changing batch size
    # cf.init_lr = cf.init_lr * args.batch_size / 64

    # model
    logger.info("creating model")
    net = model.Net(cf, logger)
    torch.cuda.set_device(local_rank)
    net.cuda(local_rank)

    # distributed model
    if args.distributed:
        args.batch_size = int(args.batch_size / args.ngpus)
        args.workers = int((args.workers + args.ngpus - 1) / args.ngpus)
        # sync_batchnorm and DistributedDataParallel
        net = SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], find_unused_parameters=True)

    if args.fix_lr:
        if hasattr(net, 'module'):
            optim_params = [{'params': net.module.ngrowth.parameters(), 'fix_lr': False},
                            {'params': net.module.fgrowth.parameters(), 'fix_lr': False},
                            {'params': net.module.perceptual_loss.parameters(), 'fix_lr': True},
                            ]
        else:
            optim_params = [{'params': net.ngrowth.parameters(), 'fix_lr': False},
                            {'params': net.fgrowth.parameters(), 'fix_lr': False},
                            {'params': net.perceptual_loss.parameters(), 'fix_lr': True},
                            ]

    else:
        optim_params = net.parameters()

    #optimizer
    if cf.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(optim_params, lr=cf.init_lr, momentum=0.9, weight_decay=cf.weight_decay, nesterov=True)
    elif cf.optimizer.lower() == 'adam':
        optimizer = optim.Adam(optim_params, lr=cf.init_lr, betas=(0.5, 0.999))
    else:
        raise ValueError('optimizer must be one of `SGD` or `Adam` or `RAdam`.')  
    
    #resume
    starting_epoch = 0
    trained_epochs = []
    if args.resume and os.path.exists(cf.checkpoint_path):
        if args.load_checkpoint_only:
            logger.info('load checkpoint only.')
            exp_utils.load_checkpoint_only(cf.checkpoint_path, net)
        else:
            starting_epoch = exp_utils.load_checkpoint(cf.checkpoint_path, net, optimizer=optimizer)
            logger.info('resumed to checkpoint {} at epoch {}'.format(cf.checkpoint_path, starting_epoch-1))

    #cuda
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # dataset
    logger.info('loading dataset and initializing batch generators...')
    dataloaders = {}
    train_dataset = dataset.DataCustom(cf, logger, phase='train')
    val_dataset = dataset.DataCustom(cf, logger, phase='val')

    if args.distributed:
        train_sampler  = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloaders['train'] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    dataloaders['val'] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)
    
    # prepare monitoring
    if not args.distributed or (args.distributed and args.rank == 0): 
        monitor_metrics = eval_utils.prepare_monitoring()
        tensorboard_writer = SummaryWriter(cf.log_dir)
    else:
        monitor_metrics, tensorboard_writer = None, None
    
    if args.only_val: 
        validate(local_rank, dataloaders, net, optimizer, starting_epoch, tensorboard_writer, monitor_metrics, args, trained_epochs, logger, cf)
        return None

    # exp_utils.save_model(net, optimizer, -1, cf.out_models)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cf.num_epochs)
    for epoch in range(starting_epoch, cf.num_epochs):
        if args.distributed:
            # different sampler per
            train_sampler.set_epoch(epoch)
        
        if epoch==0 and (not args.distributed or (args.distributed and args.rank == 0)): 
            # val
            if cf.do_validation and (not epoch % cf.validataion_per_epoch):
                logger.info("starting valiadation.")
                validate(local_rank, dataloaders, net, optimizer, epoch, tensorboard_writer, monitor_metrics, args, trained_epochs, logger, cf)

        logger.info('starting training epoch {}'.format(epoch))
        # lr adjustment
        # scheduler.step()
        if hasattr(cf, 'lr_adjustment_strategy'):
            if cf.lr_adjustment_strategy == 'consine':
                lr_utils.adjust_lr_cos(optimizer, epoch, cf.num_epochs, cf.warm_epoch, cf.init_lr, cf.fix_lr)
            elif cf.lr_adjustment_strategy == 'step':
                for param_group in optimizer.param_groups:
                    if 'fix_lr' in param_group and param_group['fix_lr']:
                        param_group['lr'] = cf.fix_lr
                    else:
                        param_group['lr'] = cf.learning_rate[epoch]
            elif cf.lr_adjustment_strategy == 'constant':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cf.init_lr
            else:
                raise ValueError('not implementation: {}'.format(cf.lr_adjustment_strategy))
        else:
            lr_utils.adjust_lr_cos(optimizer, epoch, cf.num_epochs, cf.warm_epoch, cf.init_lr, cf.fix_lr)

        if local_rank == 0:
            tensorboard_writer.add_scalar('learning_rate0', optimizer.param_groups[0]['lr'], epoch)
            tensorboard_writer.add_scalar('learning_rate-1', optimizer.param_groups[-1]['lr'], epoch)

        # train
        train(local_rank, dataloaders, net, optimizer, epoch, tensorboard_writer, monitor_metrics, args, logger, cf)

        if not args.distributed or (args.distributed and args.rank == 0): 
            # val
            if cf.do_validation and (not epoch % cf.validataion_per_epoch):
                logger.info("starting valiadation.")
                validate(local_rank, dataloaders, net, optimizer, epoch, tensorboard_writer, monitor_metrics, args, trained_epochs, logger, cf)

            if not epoch % 1:
                # update monitoring and prediction plots
                eval_utils.update_tensorboard(monitor_metrics, epoch, tensorboard_writer, cf.do_validation)

    if not args.distributed or (args.distributed and args.rank == 0):
        tensorboard_writer.close()

    if args.distributed:
        cleanup()


def train(local_rank, dataloaders, net, optimizer, epoch, tensorboard_writer, monitor_metrics, args, logger=None, cf=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    records = {}

    device = torch.device('cuda:{}'.format(local_rank) if torch.cuda.is_available() else "cpu")
    train_step = epoch * len(dataloaders['train'])

    net.train()
    end = time.time()
    for batchidx, batch_inputs in enumerate(dataloaders['train']):
        # measure data loading time
        data_time.update(time.time() - end)

        weight_norm = 0
        for module_param_name, value in net.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            weight_norm += torch.norm(value)
        args.weight_norm = weight_norm

        # control gpu
        if hasattr(cf, 'rest_time'):
            time.sleep(cf.rest_time)
        # to cuda
        inputs, targets = batch_inputs['inputs'], batch_inputs['targets']
        label = batch_inputs['label']
        targets = tensor_to_cuda(targets, device)
        inputs = tensor_to_cuda(inputs, device)
        
        # run model
        result, feat_dict = net(inputs, targets, 'train', epoch)
        loss = result['loss']
        loss = loss.mean()

        # measure accuracy and record loss
        if batchidx == 0:
            meters = [batch_time]
            for key in result.keys():
                records[key] = AverageMeter(key, ':.3f')
                meters.append(records[key])

            progress = ProgressMeter(len(dataloaders['train']), meters, 
                                    prefix="Epoch: [{}]".format(epoch), logger=logger)
        
        inputs = batch_inputs['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        bs = inputs.size(0)
        for key in result.keys():
            records[key].update(result[key].mean().item(), bs)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        _ = clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.distributed or (args.distributed and args.rank == 0):
            if batchidx % args.print_freq == 0:
                progress.display(batchidx)
                tensorboard_writer.add_scalar('train_loss_steps', loss.item(), train_step)
                tensorboard_writer.add_scalar('norm_weight', weight_norm.item(), train_step)

        train_step += 1

    if not args.distributed or (args.distributed and args.rank == 0):
        eval_utils.update_tensorboard_image(feat_dict, epoch, tensorboard_writer, name='train')
        progress.display(batchidx)
        train_loss = records['loss'].avg
        exp_utils.save_latest_model(cf, net, optimizer, epoch)
        for key in records.keys():
            if key not in monitor_metrics['train']:
                monitor_metrics['train'][key] = []
            monitor_metrics['train'][key].append(records[key].avg)

        logger.info('trained epoch {}: took {} sec.'.format(epoch, batch_time.sum))


def validate(local_rank, dataloaders, net, optimizer, epoch, tensorboard_writer, monitor_metrics, args, trained_epochs, logger=None, cf=None):
    batch_time = AverageMeter('Time', ':6.3f')
    records = {}
    device = torch.device('cuda:{}'.format(local_rank))
    net.eval()
    with torch.no_grad():
        end = time.time()
        for batchidx, batch_inputs in enumerate(dataloaders['val']):
            #to cuda
            inputs, targets = batch_inputs['inputs'], batch_inputs['targets']
            label = batch_inputs['label']
            inputs = tensor_to_cuda(inputs, device)
            targets = tensor_to_cuda(targets, device)
            result, feat_dict = net(inputs, targets, 'val', epoch)

            # measure accuracy and record loss
            if batchidx == 0:
                meters = [batch_time]
                for key in result.keys():
                    records[key] = AverageMeter(key, ':.3f')
                    meters.append(records[key])

                progress = ProgressMeter(len(dataloaders['val']), meters, 
                                        prefix='Validate: ', logger=logger)

            inputs = batch_inputs['inputs']
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            bs = inputs.size(0)
            for key in result.keys():
                records[key].update(result[key].mean().item(), bs)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.distributed or (args.distributed and args.rank == 0):
                if batchidx % args.print_freq == 0:
                    progress.display(batchidx)

    if not args.distributed or (args.distributed and args.rank == 0):
        eval_utils.update_tensorboard_image(feat_dict, epoch, tensorboard_writer, name='val')
        progress.display(batchidx)
        for key in records.keys():
            if key not in monitor_metrics['val']:
                monitor_metrics['val'][key] = []
            monitor_metrics['val'][key].append(records[key].avg)

        trained_epochs.append(epoch)
        start = len(trained_epochs)//3 # remove early epoch
        for key in ['acc', 'auc', 'dice_score_1']:  
            if key not in monitor_metrics['val'].keys():
                continue
            best_score = max(monitor_metrics['val'][key][start:])        
            ep = trained_epochs[start:][monitor_metrics['val'][key][start:].index(best_score)]
            logger.info('best {}: {}, ep: {}'.format(key, best_score, ep))
      
        exp_utils.model_select_save(cf, net, optimizer, monitor_metrics, epoch, trained_epochs)
        logger.info('validated epoch {}: took {} sec.'.format(epoch, batch_time.sum))


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", init_method='env://', rank=rank, world_size=world_size)
    torch.distributed.barrier()

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(20)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()