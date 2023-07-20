#encoding=utf-8
import os
import time
import glob
import math
import random
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom

from med_dataset import transforms as med_transforms


aug_params = {
    "do_spatial": False,
    "do_mirror": False,
    "do_gamma": True,
    "do_gauss": True,
    "do_color": True,
    "do_simulatelowresolution": False,
    "do_removeLabel": False,

    # spatial param
    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2, #0.2

    "border_mode_data": 'nearest',
    "border_mode_seg": 'constant',
    "border_val_data": 0,
    "border_val_seg": 0,
    "sample_order_data": 1,
    "sample_order_seg": 1,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": True,
    "p_scale": 0.2,
    "border_val_seg": 0,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_border_crop": False,
    "random_center_crop": False, # if-elif-else
    "random_crop_dist_to_border": 30,
    "random_crop_dist_to_center": 20,

    # gamma param
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3, #0.3

    # mirror axes
    "mirror_axes": (1, 2), # x,y

    # other param
    "p_gauss_noise": 0.1, #0.1
    "p_gauss_blur": 0.2, #0.2
    "blur_sigma": (0.5, 1.),

    "p_brightness": 0.15, #0.15
    "p_contrast": 0.15,
    "brightness_multiplier_range": (0.85, 1.15),
    "contrast_range": (0.85, 1.15),
    "p_simulatelowresolution": 0.25,
}


def build_transform_med(is_train, input_size=96):
    transform_train = med_transforms.Compose([ 
        med_transforms.CenterPadCrop(input_size+8, pad_value=-1024),
        med_transforms.RandomPadCrop(input_size, pad_value=-1024),
        med_transforms.RandomFlipAxis(axis=2,coord_start_axis=1), #y
        med_transforms.RandomFlipAxis(axis=3,coord_start_axis=1), #x
    ])
    transform_test = med_transforms.Compose([
        med_transforms.CenterPadCrop(input_size, pad_value=-1024),
    ])
    return transform_train if is_train else transform_test


class LnGrowthDataset(torch.utils.data.Dataset):

    def __init__(self, root, train=True, input_size=72):

        super(LnGrowthDataset, self).__init__()
        self.input_size = input_size
        self.resolution = [1.0, 1.0, 1.0]
        self.pre_crop_size = input_size + 16 # mm
        self.transform = build_transform_med(is_train=train, input_size=self.input_size)
        self.do_more_aug = False

        self.root = root
        print('Data path: ', self.root)
        self.train = train
        if self.train:
            self.phase = 'train'
            self.csv_path = [
                '/data/pair_nodules_train_dataset.csv'
            ]
        else:
            self.phase = 'val'
            self.csv_path = [
                "/data/pair_nodules_val_dataset.csv"
            ]
        
        self.data_dirs = [
            "/data_share/lung_cancer_data/NLST/nlst_processed/dataset_solid/",
            "/data_share/lung_cancer_data/NLST/nlst_processed/dataset_ssn/"
        ]
        self.mask_dir = [
            "/data_share/lung_cancer_data/NLST/nlst_processed/NLST_Solid_mask/",
            "/data_share/lung_cancer_data/NLST/nlst_processed/NLST_SSN_mask/"
        ]

        self.path_list = []
        if self.csv_path is not None:
            for di, data_dir in enumerate(self.data_dirs):
                for csv_path_s in self.csv_path:
                    df = pd.read_csv(csv_path_s)
                    for i in range(len(df)):
                        uid = df.loc[i,'id']
                        cli_info = df.loc[i,['age', 'gender']].values
                        diams = df.loc[i,['Seg_diam0', 'Seg_diam1', 'Seg_diam2']].values
                        paths = df.loc[i,['path0', 'path1', 'path2']].values
                        data = {'cli_info':cli_info, 'uid': uid, 'days':df.loc[i,'days0':'days2'].values,
                                'label': df.loc[i,'label']
                        }

                        for i, path in enumerate(paths):
                            if type(path)==str and path!='':
                                path = os.path.join(data_dir, path)
                                fn = os.path.splitext(os.path.basename(path))[0]
                                mask_path = os.path.join(self.mask_dir[di], fn+'_mask.nii.gz')
                            
                                if os.path.exists(path) and os.path.exists(mask_path):                     
                                    data['path%d'%i] = path
                                    data['mask_path%d'%i] = mask_path
                                else:
                                    data['path%d'%i] = None
                                    data['mask_path%d'%i] = None
                            else:
                                data['path%d'%i] = None
                                data['mask_path%d'%i] = None

                        if (data['path1'] is not None) and ((data['path0'] is not None) or (data['path2'] is not None)):
                            self.path_list.append(data)
                        # self.path_list.append(data)
        
        print('Num of data: {}'.format(len(self.path_list)))
        
    
    def __len__(self):
        return len(self.path_list)

    def _transform(self, image_zyx, mask_zyx, transform, transforms_more=None):
        inputs = {'data': image_zyx, 'mask':mask_zyx}
        inputs = transform(**inputs)
        image_zyx, mask_zyx = inputs['data'].copy(), inputs['mask'].copy()
        # normalize
        vmin, vmax = -1024, 1000
        image_zyx = np.clip(image_zyx, vmin, vmax)
        image_zyx = image_zyx.astype(np.float32)
        image_zyx = (image_zyx-vmin)/(vmax-vmin)
        # image_zyx = image_zyx*2.0 - 1.0
        # mean, std = -475.7, 1000
        # image_zyx = np.clip(image_zyx, -1024, 600)
        # image_zyx = image_zyx.astype(np.float32)
        # image_zyx = (image_zyx-mean)/std

        if self.do_more_aug and transforms_more is not None:
            inputs = {'data':image_zyx[None], 'seg':mask_zyx[None]}
            inputs = transforms_more(**inputs)
            image_zyx, mask_zyx = inputs['data'].squeeze(), inputs['seg'].squeeze()
        return image_zyx, mask_zyx

    def __getitem__(self, index):
        data = self.path_list[index]
        path0, path1, path2 = data['path0'], data['path1'], data['path2']
        mask_path0, mask_path1, mask_path2 = data['mask_path0'], data['mask_path1'], data['mask_path2']
        day0, day1, day2 = data['days']
        fn = data['uid']
        label = data['label']
        if label == 2:
            label = 1
        shift = 0
        if not isinstance(path2, str):
            shift = 1
            path0, path1, path2 = path0, path0, path1
            day0, day1, day2 = day0, day0, day1
            mask_path0, mask_path1, mask_path2 = mask_path0, mask_path0, mask_path1
            
        if not isinstance(path0, str):
            shift = 1
            path0, day0, mask_path0 = path1, day1, mask_path1
        
        paths = [path0, path1, path2]
        mask_paths = [mask_path0, mask_path1, mask_path2]
        days = [day0, day1, day2]

        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))
        if self.phase=='train' and shift==0:
            cur_i = np.random.randint(2)
            next_i = np.random.randint(cur_i+1, high=3)
        else:
            cur_i, next_i = 1, 2
        
        path0, path1 = paths[cur_i], paths[next_i]
        mask_path0, mask_path1 = mask_paths[cur_i], mask_paths[next_i]
        day0, day1 = days[cur_i], days[next_i]

        ddays = day1 - day0  
        # tt = ddays/365.0  # normalize day, year
        tt = ddays/30.0     # normalize day, month
        assert tt>0, 'dd must greater than 0'

        gt_gimage, mask_zyx = self.load_image_transform(path0, path1, mask_path0, mask_path1)
        image_zyx0, image_zyx1 = gt_gimage
        mask_zyx0, mask_zyx1 = mask_zyx

        target_image, target_mask = image_zyx1[None], mask_zyx1[None]

        if self.phase=='train' and np.random.rand()<0.2:
            tt = 0
            target_image, target_mask = image_zyx0[None], mask_zyx0[None]
        
        lnm = np.log((target_image*target_mask).sum()/(image_zyx0*mask_zyx0).sum())
        lnv = np.log(target_mask.sum()/mask_zyx0.sum())
        
        v0 = mask_zyx0.sum()
        m0 = (image_zyx0*mask_zyx0).sum()
        v1 = target_mask.sum()
        m1 = (target_image*target_mask).sum()

        # if self.phase=='train':
        #     lnm = lnm + np.random.normal(0, 0.05)
        #     lnv = lnv + np.random.normal(0, 0.05)
        
        return {
            'inputs': [image_zyx0[None], mask_zyx0[None], v0, m0, v1, m1, tt], 
            'targets': [target_image, target_mask, lnv, lnm, label], 
            'uid': data['uid'],
            }



    def load_image_transform(self, path0, path1, mask_path0=None, mask_path1=None):
        # load t0
        image_zyx0 = np.load(path0)['img']
        if mask_path0 is not None:
            sitk_mask0 = sitk.ReadImage(mask_path0)
            mask_zyx0 = sitk.GetArrayFromImage(sitk_mask0)
        else:
            mask_zyx0 = np.zeros_like(image_zyx0)
        # load t1, target image
        image_zyx1 = np.load(path1)['img']
        ori_shape = image_zyx1.shape
        spacing_zyx = np.array([1.0,1.0,1.0])
        if mask_path1 is not None:
            sitk_mask1 = sitk.ReadImage(mask_path1)
            mask_zyx1 = sitk.GetArrayFromImage(sitk_mask1)
        else:
            mask_zyx1 = np.zeros_like(image_zyx1)

        # concat
        image_zyx = np.stack([image_zyx0, image_zyx1], axis=0)
        mask_zyx = np.stack([mask_zyx0, mask_zyx1], axis=0)

        # transform
        pre_crop_size =  self.pre_crop_size/np.array(spacing_zyx)
        assert pre_crop_size[0]<=ori_shape[0] and pre_crop_size[1]<=ori_shape[1] and pre_crop_size[2]<=ori_shape[2], 'Pre-crop size must be smaller than ori-shape' 
        crop_func = med_transforms.CenterPadCrop(pre_crop_size, pad_value=-1024, dims=3)
        inputs = crop_func(data=image_zyx, mask=mask_zyx, centerd=None)
        image_zyx, mask_zyx = inputs['data'], inputs['mask']

        sp_scale = np.array(spacing_zyx)/self.resolution
        sp_scale = np.insert(sp_scale, 0, 1.0)
        image_zyx = zoom(image_zyx, sp_scale, order=1).astype(np.float32)
        mask_zyx = zoom(mask_zyx, sp_scale, order=0).astype(np.float32)

        image_zyx, mask_zyx = self._transform(image_zyx, mask_zyx, self.transform)

        return image_zyx, mask_zyx

    
