#encoding=utf-8
import argparse
import os
import time
import torch
import numpy  as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import csv

from utils.logger import get_logger
import utils.exp_utils as exp_utils
from utils.convert import tensor_to_cuda

import warnings
warnings.filterwarnings("ignore")



# 仅用于验证
def val(logger, cf, model, dataset):

    logger.info("performing training with model {}".format(cf.model))

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    net = model.Net(cf, logger)
    net.to(device)

    if os.path.exists(cf.checkpoint_path):
        resume_epoch = exp_utils.load_checkpoint(cf.checkpoint_path, net, optimizer=None)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.checkpoint_path, resume_epoch))
    else:
        logger.info('No checkpoint!')
    # 多GPU并行训练
    if torch.cuda.device_count() > 1:
        net = DataParallel(net)
    
    #add this , can improve the train speed
    torch.backends.cudnn.benchmark = True
    logger.info('loading dataset and initializing batch generators...')
    
    with torch.no_grad():
        val_dataset = dataset.DataCustom(cf, logger=None, phase='infer')
        dataloaders = {}
        dataloaders['val'] = DataLoader(val_dataset,
                                        batch_size=cf.batch_size,
                                        shuffle=False,
                                        num_workers=cf.n_workers,
                                        pin_memory=True)
        logger.info("starting inference.")

        dice_list, PSNR, SSIM, mse_list, msenod_list, msenod2_list = [], [], [], [], [], []
        net.eval()
        toc = time.time()
        for batchidx, batch_inputs in enumerate(dataloaders['val']):
            print ('inference: {}/{}'.format(batchidx, len(dataloaders['val'])))
            tic_fw = time.time()
            inputs, targets, transform, label = batch_inputs['inputs'], batch_inputs['targets'], batch_inputs['transform'], batch_inputs['label']
            uid = batch_inputs['uid']
            inputs = tensor_to_cuda(inputs, device)
            targets = tensor_to_cuda(targets, device)
            image_zyx0, mask_zyx0, tt = inputs
            target_image, target_mask, lnv, lnm = targets

            pred, dice, mse, mse_nod, mse_nod2, psnr, ssim = net(inputs, targets, transform=transform, phase='infer')

            dice_list.append(dice.cpu().numpy().squeeze())
            PSNR.append(psnr)
            SSIM.append(ssim)
            mse_list.append(mse.cpu().numpy().squeeze())
            msenod_list.append(mse_nod.cpu().numpy().squeeze())
            if mse_nod2!=0:
                msenod2_list.append(mse_nod2.cpu().numpy().squeeze())

        print ('dice:{0}\n'.format(np.mean(np.array(dice_list))))
        print ('SSIM:{0}\n'.format(np.mean(np.array(SSIM))))

        print('PSNR:{0} \n MSE:{1} \n MSE_nod:{2} \n MSE_nod2:{3}'.format(np.mean(np.array(PSNR)), np.mean(np.array(mse_list)), np.mean(np.array(msenod_list)), np.mean(np.array(msenod2_list))))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='infer',
                        help='one out of : train / infer / train_infer')
    parser.add_argument('--gpu', type=str, default='3',
                        help='assign which gpu to use.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='train batch size.')
    parser.add_argument('--database', action='store_true', default=False,
                        help='train batch size.')
    parser.add_argument('--use_stored_settings', action='store_true', default=True,
                        help='load configs from existing stored_source instead of exp_source. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--exp_source', type=str, default='experiments/',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    parser.add_argument('--stored_source', type=str, default='./output/nodule_generation/sources',
                        help='specifies, from which source experiment to load configs, data_loader and model.')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cf = exp_utils.prep_exp(args.exp_source, args.stored_source, args.use_stored_settings, is_train=False)
    
    if args.batch_size > 0:
        cf.batch_size = args.batch_size
    else:
        cf.batch_size = 1
    
    cf.database = args.database
    cf.n_workers = 8

    model = exp_utils.import_module('model', cf.model_path)
    dataset = exp_utils.import_module('dataset', cf.dataset_path)

    logger = get_logger(cf.log_dir)
    val(logger, cf, model, dataset)