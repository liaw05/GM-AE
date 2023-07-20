import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import os
import time
import torch
from tqdm import tqdm
from torchvision.utils import make_grid


def prepare_monitoring():
    """
    creates dictionaries, where train/val metrics are stored.
    """
    metrics = {'train': {}, 'val': {}}
    return metrics


def update_tensorboard(metrics, epoch, tensorboard_writer, do_validation=True):
    keys = metrics['train'].keys()
    scalar_dict_train = {}
    scalar_dict_val = {}
    for i, key in enumerate(keys):
        scalar_dict_train[key] = metrics['train'][key][-1]
        if do_validation and key in metrics['val']:
            scalar_dict_val[key] = metrics['val'][key][-1]
            
    tensorboard_writer.add_scalars('train', scalar_dict_train, epoch)
    tensorboard_writer.add_scalars('val', scalar_dict_val, epoch)


def update_tensorboard_image(feat_dict, global_step, tensorboard_writer, name='train'):
    for i, key in enumerate(feat_dict):
        if not('pred' in key):
            feat_map = feat_dict[key]
            tensorboard_writer.add_image(name+'_'+key, make_grid(feat_map,
                                        padding=20, normalize=True,
                                        scale_each=True, pad_value=1), global_step)


def update_tensorboard_fmap(featuremaps, tensorboard_writer, epoch):
    grad_dict = {}
    for i, fmap in enumerate(featuremaps):
        if isinstance(featuremaps, dict):
            grad_dict[i] = featuremaps[fmap]
        else:
            grad_dict[i] = fmap
        tensorboard_writer.add_histogram('fmap%d'%i, grad_dict[i], epoch)
    
    # torch.save({'state_dict': grad_dict},
    #     os.path.join('/data/code/Med3D-SSPTNet/output/20210827ln_mal_cls_res34_temp/out_models/featmap0.tar'))


def update_tensorboard_weight(net, tensorboard_writer, epoch):
    visual_names = [
        'encoder_q.C1.weight',
        'encoder_q.conv_list.0.1.conv1.weight', 'encoder_q.conv_list.0.1.conv2.weight', 
        'encoder_q.conv_list.0.2.conv1.weight', 'encoder_q.conv_list.0.2.conv2.weight', 
        'encoder_q.conv_list.1.0.conv1.weight', 'encoder_q.conv_list.1.0.conv2.weight', 
        'encoder_q.conv_list.1.3.conv1.weight', 'encoder_q.conv_list.1.3.conv2.weight', 
        'encoder_q.conv_list.2.0.conv1.weight', 'encoder_q.conv_list.2.0.conv2.weight', 
        'encoder_q.conv_list.2.4.conv1.weight', 'encoder_q.conv_list.2.4.conv2.weight', 
        'encoder_q.conv_list.2.5.conv1.weight', 'encoder_q.conv_list.2.5.conv2.weight', 
        'encoder_q.conv_list.3.0.conv1.weight', 'encoder_q.conv_list.3.0.conv2.weight', 
        'encoder_q.conv_list.3.2.conv1.weight', 'encoder_q.conv_list.3.2.conv2.weight', 
        'encoder_q.down_tr64.ops.0.conv1.weight',
        'encoder_q.down_tr128.ops.0.conv1.weight','encoder_q.down_tr128.ops.1.conv1.weight',
        'encoder_q.down_tr256.ops.0.conv1.weight','encoder_q.down_tr256.ops.1.conv1.weight',
        'encoder_q.down_tr512.ops.0.conv1.weight','encoder_q.down_tr512.ops.1.conv1.weight',
        'encoder_q.conv1.weight',
        'encoder_q.layer1.2.conv1.weight','encoder_q.layer1.2.conv2.weight',
        'encoder_q.layer2.3.conv1.weight','encoder_q.layer2.3.conv2.weight',
        'encoder_q.layer3.5.conv1.weight','encoder_q.layer3.5.conv2.weight',
        'encoder_q.layer4.2.conv1.weight','encoder_q.layer4.2.conv2.weight',
    ]
    grad_dict = {}
    for name, param in net.named_parameters():
        if name in visual_names:
            tensorboard_writer.add_histogram(name+'_data', param.data.detach(), epoch)
            if param.grad is not None:
                grad_dict[name] = param.grad
                tensorboard_writer.add_histogram(name+'_grad', param.grad, epoch)

    # torch.save({'state_dict': grad_dict},
    #     os.path.join('/data/code/Med3D-SSPTNet/output/20210827ln_mal_cls_res34_temp/out_models/grad0.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        if logger is None:
            self.printf = print
        else:
            self.printf = logger.info

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.printf('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'