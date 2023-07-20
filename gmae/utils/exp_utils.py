import os
import sys
import csv
import logging
import subprocess
import glob

import torch
import numpy as np
import pandas as pd
import importlib
from collections import OrderedDict


def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prep_exp(exp_source, stored_source, use_stored_settings=False, is_train=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    if use_stored_settings:
        cf_file = import_module('cf', os.path.join(stored_source, 'configs.py'))
        cf = cf_file.cf
        cf.model_path = os.path.join(stored_source, 'model.py')
        cf.dataset_path = os.path.join(stored_source, 'dataset.py')

    else:
        cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
        cf = cf_file.cf
        cf.dataset_path = os.path.join(exp_source, 'dataset.py')
        cf.model_path = 'models/{}.py'.format(cf.model)

    if is_train and not use_stored_settings:
        if not os.path.exists(cf.save_sources):
            os.makedirs(cf.save_sources)
        subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(cf.save_sources, 'model.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'configs.py'), os.path.join(cf.save_sources, 'configs.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'dataset.py'), os.path.join(cf.save_sources, 'dataset.py')), shell=True)

    if not os.path.exists(cf.log_dir):
        os.makedirs(cf.log_dir)
    if not os.path.exists(cf.out_models):
        os.makedirs(cf.out_models)

    return cf


def prep_exp_ddp(exp_source, stored_source, use_stored_settings=False, is_train=True):
    """
    I/O handling, creating of experiment folder structure. Also creates a snapshot of configs/model scripts and copies them to the exp_dir.
    This way the exp_dir contains all info needed to conduct an experiment, independent to changes in actual source code. Thus, training/inference of this experiment can be started at anytime. Therefore, the model script is copied back to the source code dir as tmp_model (tmp_backbone).
    Provides robust structure for cloud deployment.
    :param dataset_path: path to source code for specific data set. (e.g. medicaldetectiontoolkit/lidc_exp)
    :param exp_path: path to experiment directory.
    :param use_stored_settings: boolean flag. When starting training: If True, starts training from snapshot in existing experiment directory, else creates experiment directory on the fly using configs/model scripts from source code.
    :return:
    """

    if use_stored_settings:
        cf_file = import_module('cf', os.path.join(stored_source, 'configs.py'))
        cf = cf_file.cf
        cf.save_sources = stored_source
        cf.model_path = os.path.join(stored_source, 'model.py')
        cf.dataset_path = os.path.join(stored_source, 'dataset.py')

    else:
        cf_file = import_module('cf', os.path.join(exp_source, 'configs.py'))
        cf = cf_file.cf
        cf.dataset_path = os.path.join(exp_source, 'dataset.py')
        cf.model_path = 'models/{}.py'.format(cf.model)

    if is_train and not use_stored_settings:
        if not os.path.exists(cf.save_sources):
            os.makedirs(cf.save_sources)
        subprocess.call('cp {} {}'.format(cf.model_path, os.path.join(cf.save_sources, 'model.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'configs.py'), os.path.join(cf.save_sources, 'configs.py')), shell=True)
        subprocess.call('cp {} {}'.format(os.path.join(exp_source, 'dataset.py'), os.path.join(cf.save_sources, 'dataset.py')), shell=True)

    if not os.path.exists(cf.log_dir):
        os.makedirs(cf.log_dir)
    if not os.path.exists(cf.out_models):
        os.makedirs(cf.out_models)

    return cf


def model_select_save(cf, net, optimizer, monitor_metrics, epoch, trained_epochs):
    keys = monitor_metrics['train'].keys()
    for key in ['auc_v', 'auc', 'acc', 'dice', 'c-index', 'loss']:
        if key in keys:
            break
    if cf.do_validation:
        val_losses = monitor_metrics['val'][key]
    else:
        val_losses = monitor_metrics['train'][key]

    val_losses = np.array(val_losses)
    index_ranking = np.argsort(val_losses) if key=='loss' else np.argsort(-val_losses)
    epoch_ranking = np.array(trained_epochs)[index_ranking]
    
    # check if current epoch is among the top-k epchs.
    if epoch in epoch_ranking[:cf.save_num_best_models]:
        select_model_dir = os.path.join(cf.out_models, 'select_model')
        if not os.path.exists(select_model_dir):
            os.makedirs(select_model_dir)
        save_model(net, optimizer, epoch, select_model_dir)
        # 更新为最佳模型地址
        cf.checkpoint_path = os.path.join(select_model_dir, 'model_%03d.tar' % epoch)

        # delete params of the epoch that just fell out of the top-k epochs.
        if len(epoch_ranking) > cf.save_num_best_models:
            epoch_rm = epoch_ranking[cf.save_num_best_models]
            subprocess.call('rm {}'.format(os.path.join(select_model_dir, 'model_%03d.tar' % epoch_rm)), shell=True)


def save_latest_model(cf, net, optimizer, epoch):
    latest_model_dir = os.path.join(cf.out_models, 'latest_model')
    if not os.path.exists(latest_model_dir):
        os.makedirs(latest_model_dir)
    # if not epoch % cf.save_model_per_epochs:
    #     save_model(net, optimizer, epoch, cf.out_models)
    
    # save lastest checkpoints
    if hasattr(cf, 'save_num_latest_models'):     
        save_model(net, optimizer, epoch, latest_model_dir)
        save_epoch = [int(fn[:-4].split('_')[-1]) for fn in os.listdir(latest_model_dir)]
        if len(save_epoch) > cf.save_num_latest_models:       
            save_epoch.sort(reverse=False)
            epoch_rm = save_epoch[0]
            subprocess.call('rm {}'.format(os.path.join(latest_model_dir, 'model_%03d.tar' % epoch_rm)), shell=True)


def save_model(net, optimizer, epoch, model_dir):
    if hasattr(net, 'module'):
        net_state_dict = net.module.state_dict()
    else:
        net_state_dict = net.state_dict()

    torch.save({
        'epoch': epoch+1,
        'state_dict': net_state_dict,
        'optimizer': optimizer.state_dict(),
        }, 
        os.path.join(model_dir, 'model_%03d.tar' % epoch),
        _use_new_zipfile_serialization=False)


def load_checkpoint(checkpoint_path, net, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    try:
        if hasattr(net, 'module'):
            pretrain_dict = {'module.'+k:v for k,v in pretrain_dict.items()}
        net.load_state_dict(pretrain_dict)
        print('***load parameters completely***')
    except:
        print('***load part parameters***')
        model_dict = net.state_dict()
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
        print('Num of param:', len(pretrain_dict))
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']


def load_checkpoint_only(checkpoint_path, net):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    try:
        if hasattr(net, 'module'):
            pretrain_dict = {'module.'+k:v for k,v in pretrain_dict.items()}
        net.load_state_dict(pretrain_dict)
        print('***load parameters completely***')
    except:
        print('***load part parameters***')
        model_dict = net.state_dict()
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
        print('Num of param:', len(pretrain_dict))
        model_dict.update(pretrain_dict)
        net.load_state_dict(model_dict)


def load_checkpoint_ema(checkpoint_pathes, net, m=0.9):
    model_dict = net.state_dict()
    pretrained_dict_ema = {}
    if isinstance(checkpoint_pathes, str):
        checkpoint_pathes = glob.glob(os.path.join(checkpoint_pathes,'*.tar'))
    print('Num of checkpoints: ', len(checkpoint_pathes))

    save_epoch = [(fn, int(os.path.basename(fn)[:-4].split('_')[-1])) for fn in checkpoint_pathes]
    save_epoch.sort(key=lambda x: x[1], reverse=False)
    for checkpoint_path, _ in save_epoch:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        for k, v in pretrained_dict.items():
            if k in pretrained_dict_ema:
                pretrained_dict_ema[k] = pretrained_dict_ema[k]*m + v*(1-m)
            else:
                pretrained_dict_ema[k] = v

    model_dict.update(pretrained_dict_ema)
    net.load_state_dict(model_dict)


def load_checkpoint_agv(checkpoint_pathes, net):
    model_dict = net.state_dict()
    pretrained_dict_ema = {}
    if isinstance(checkpoint_pathes, str):
        checkpoint_pathes = glob.glob(os.path.join(checkpoint_pathes,'*.tar'))
    print('Num of checkpoints: ', len(checkpoint_pathes))

    for checkpoint_path in checkpoint_pathes:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        for k, v in pretrained_dict.items():
            if k in pretrained_dict_ema:
                pretrained_dict_ema[k] += v/len(checkpoint_pathes)
            else:
                pretrained_dict_ema[k] = v/len(checkpoint_pathes)

    model_dict.update(pretrained_dict_ema)
    net.load_state_dict(model_dict)


def merge_checkpoint_agv(checkpoint_pathes, save_dir):
    pretrained_dict_agv = {}
    if isinstance(checkpoint_pathes, str):
        checkpoint_pathes = glob.glob(os.path.join(checkpoint_pathes,'*.tar'))
    print('Num of checkpoints: ', len(checkpoint_pathes))

    for checkpoint_path in checkpoint_pathes:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_dict = checkpoint['state_dict']
        for k, v in pretrained_dict.items():
            if k in pretrained_dict_agv:
                pretrained_dict_agv[k] += v/len(checkpoint_pathes)
            else:
                pretrained_dict_agv[k] = v/len(checkpoint_pathes)

    torch.save({
        'state_dict': pretrained_dict_agv,},
        os.path.join(save_dir, 'model_agv.tar'))


def high_to_low_version(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    save_dict = {}
    for k in checkpoint.keys():
        save_dict[k] = checkpoint[k]
    torch.save(save_dict,
        checkpoint_path[:-4]+'_low.tar', _use_new_zipfile_serialization=False)


def clean_pretrained_weight(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')   
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = {k[19:]:v for k,v in pretrained_dict.items() if k[:19]=='encoder_q.backbone.'}
    print('Num of pretrained params: ', len(pretrained_dict))
    torch.save(pretrained_dict,
        checkpoint_path[:-4]+'.pth', _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    checkpoint_path = '/data/code/Med3D-SSPTNet/output/exp_sspt_det_mr2d/out_models/latest_model/model_050.tar'
    # high_to_low_version(checkpoint_path)
    clean_pretrained_weight(checkpoint_path)