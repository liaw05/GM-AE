# coding=utf-8
import os
from easydict import EasyDict as edict


cf = edict()
#model setting
cf.model = 'GMAE' 
cf.mode_dl= 'segmentation' # 'segmentation', 'classification', 'detection'

#save dir
cf.output_dir = './output/nodule_generation/'

# train and val data
cf.data_dirs = [
    "/data_share/lung_cancer_data/NLST/nlst_processed/dataset_solid/",
    "/data_share/lung_cancer_data/NLST/nlst_processed/dataset_ssn/"
]
cf.mask_dirs = [
    "/data_share/lung_cancer_data/NLST/nlst_processed/NLST_Solid_mask/",
    "/data_share/lung_cancer_data/NLST/nlst_processed/NLST_SSN_mask/"
]


cf.train_csv_paths = [
    "/data/pair_nodules_train_dataset.csv"
]
cf.val_csv_paths = ['/data/pair_nodules_val_dataset.csv']

cf.rate_csv = '/data_local/code/Lung_cancer_NLST/growth_image_prediction/csv/predAll.csv'

# infer data
cf.infer_data_dirs = cf.data_dirs
cf.infer_csv_paths = cf.val_csv_paths

# resume checkpoint path
cf.checkpoint_path = 'pretrained_models/gmae_base.tar'

#train
cf.num_epochs = 100
cf.weight_decay = 1e-4
cf.save_num_best_models = 100
cf.save_num_latest_models = 100
cf.save_model_per_epochs = 100
cf.validataion_per_epoch = 2
cf.warm_epoch = 4
cf.momentum = 0.8
cf.rest_time = 0.1
cf.do_validation = True
cf.infer_zoom_crop_order = False
cf.do_more_aug = False

# parameters
#zyx
cf.num_classes = 2
cf.input_size = 72 #mm
cf.resolution = [1.0, 1.0, 1.0]
cf.crop_size = [64, 64, 64]

cf.input_size_patch = 36 #mm
cf.resolution_patch = [0.5,0.5,0.5]
cf.crop_size_patch = [32, 32, 32]

cf.optimizer = 'SGD'  # SGD
cf.init_lr = 1e-2
cf.fix_lr = 1e-4
cf.lr_adjustment_strategy = 'consine' #'step','consine','constant'

# step lr
if cf.lr_adjustment_strategy == 'step':
    learning_rate = []
    for i in range(cf.num_epochs):
        if i <= 10:
            learning_rate.append(cf.init_lr*0.1)
        elif i <= cf.num_epochs*0.5:
            learning_rate.append(cf.init_lr)
        elif i <= cf.num_epochs*0.9:
            learning_rate.append(cf.init_lr * 0.1)
        else:
            learning_rate.append(cf.init_lr * 0.02)

    cf.learning_rate = learning_rate

#data augment
cf.rd_scale = (0.8, 1.2)

#save dir
cf.infer_mask = os.path.join(cf.output_dir, 'mask')
cf.log_dir = os.path.join(cf.output_dir, 'logs')
cf.out_models = os.path.join(cf.output_dir, 'out_models')
cf.save_sources = os.path.join(cf.output_dir, 'sources')

cf.is_convert_to_voxel_coord = True
cf.is_convert_to_world_coord = True # if True, convert coord to world coord, if False, convert to voxel coord.
cf.result_csv_path = os.path.join(cf.output_dir, 'results.csv')
