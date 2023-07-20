# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import pandas as pd
import torch
import csv
import numpy as np
from sklearn import metrics
from timm.data import Mixup
from timm.utils import ModelEma

from losses.distillation_losses import DistillationLoss
from utilities import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('cnn_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        samples, targets = data_dict['inputs'], data_dict['targets']
        samples = tensor_to_cuda(samples, device)
        targets = tensor_to_cuda(targets, device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            result, feat_dict = model(samples, targets, phase='train')
            loss = result['loss']
            loss_vrate = result['loss_vrate']
            loss_mrate = result['loss_mrate']

        loss_value = loss.item()
        loss_vrate_value = loss_vrate.item()
        loss_mrate_value = loss_mrate.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        if epoch != 0:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_vrate=loss_vrate_value)
        metric_logger.update(loss_mrate=loss_mrate_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # metric_logger.update(cnn_lr=optimizer.param_groups[-1]["lr"])
        result.pop('predlnv')
        result.pop('predlnm')
        for k in result:
            metric_logger.meters[k].update(result[k].item())

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion=None, num_class=2, save_csv=None, save_mask=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    predvs, predms = [], []
    gts, idlist = [], []
    gtlnv, gtlnm = [], []
    for data_dict in metric_logger.log_every(data_loader, 10, header):
        samples, targets = data_dict['inputs'], data_dict['targets']
        uid = data_dict['uid']

        samples = tensor_to_cuda(samples, device)
        targets = tensor_to_cuda(targets, device)

        target_image, target_mask, lnv, lnm, label = targets
        gts.append(label)
        gtlnv.append(lnv)
        gtlnm.append(lnm)
        idlist.append(uid)

        # compute output
        with torch.cuda.amp.autocast():
            result, feat_dict = model(samples, targets, phase='train')
            loss = result['loss']
            loss_vrate = result['loss_vrate']
            loss_mrate = result['loss_mrate']

        predvs.append(result.pop('predlnv'))
        predms.append(result.pop('predlnm'))

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_vrate=loss_vrate.item())
        metric_logger.update(loss_mrate=loss_mrate.item())
        for k in result:
            if k != 'z':
                metric_logger.meters[k].update(result[k].item())

    predvs = torch.cat(predvs, dim=0).cpu().detach().numpy()
    predms = torch.cat(predms, dim=0).cpu().detach().numpy()
    gts = (torch.cat(gts, dim=0)>0).cpu().detach().numpy()
    gtlnv = torch.cat(gtlnv, dim=0).cpu().detach().numpy()
    gtlnm = torch.cat(gtlnm, dim=0).cpu().detach().numpy()
    idlist = np.concatenate(np.array(idlist))

    auc_v = metrics.roc_auc_score(gts, predvs, average='macro')
    auc_m = metrics.roc_auc_score(gts, predms, average='macro')
    metric_logger.meters['auc_v'].update(auc_v, n=1)
    metric_logger.meters['auc'].update(auc_m, n=1)

    df = pd.DataFrame({'nid':idlist, 'label':gts, 'predlnV':predvs, 'predlnM':predms, 'lnV':gtlnv, 'lnm':gtlnm})
    df.to_csv('lngrowth_vit_gompertz.csv',index=None)

    print('loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    print('* auc_m {auc.global_avg:.3f} auc_v {auc_v.global_avg:.3f}'
        .format(auc=metric_logger.auc, auc_v=metric_logger.auc_v))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def write_csv(csv_name, content, mul=True, mod="w"):
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)


def tensor_to_cuda(tensor, device):
    if isinstance(tensor, dict):
        for key in tensor:
            tensor[key] = tensor_to_cuda(tensor[key], device)
        return tensor
    elif isinstance(tensor, (list, tuple)):
        tensor = [tensor_to_cuda(t, device) for t in tensor]
        return tensor
    else:
        return tensor.to(device)