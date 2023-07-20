#encoding=utf-8
import os
from sys import path
import time
import math

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom

from utils import transforms


class DataCustom(torch.utils.data.Dataset):

    def __init__(self, cf, logger=None, phase='train'):

        super(DataCustom, self).__init__()
        self.phase = phase
        self.cf = cf
        self.resolution = np.array(cf.resolution)
        self.data_dirs_ssn = cf.data_dirs[0]
        self.data_dirs_solid = cf.data_dirs[1]
        self.mask_dirs_ssn = cf.mask_dirs[0]
        self.mask_dirs_solid = cf.mask_dirs[1]
        self.input_size = cf.input_size
        self.crop_size = np.array(cf.crop_size)
        self.crop_size_patch = np.array(cf.crop_size_patch)
        self.do_more_aug = cf.do_more_aug
        self.rate_df = pd.read_csv(cf.rate_csv)
         
        if phase == 'train':
            self.csv_path = cf.train_csv_paths
        elif phase == 'val' or cf.infer_csv_paths == None:
            self.csv_path = cf.val_csv_paths
        else:
            self.csv_path = cf.infer_csv_paths
        
        self.path_list = []
        if self.csv_path is not None:
            for csv_path_s in self.csv_path:
                df = pd.read_csv(csv_path_s)
                for i in range(len(df)):
                    uid = df.loc[i,'id']
                    texture = df.loc[i,'Texture_label']
                    cli_info = df.loc[i,['age', 'gender']].values
                    diams = df.loc[i,['Seg_diam0', 'Seg_diam1', 'Seg_diam2']].values  
                    label = df.loc[i,'label']
                    if label == 2:
                        label = 1   
                    data = {'cli_info':cli_info, 'uid': uid, 'days':df.loc[i,'days0':'days2'].values,'label':label}

                    paths = df.loc[i,['path0', 'path1', 'path2']].values
                    for i, path in enumerate(paths):
                        if type(path)==str and path!='' and (not np.isnan(diams[i])):
                            if texture == 2:
                                path = os.path.join(self.data_dirs_solid, path.strip('/'))
                                fn = os.path.splitext(os.path.basename(path))[0]
                                mask_path = os.path.join(self.mask_dirs_solid, fn+'_mask.nii.gz')
                            else:
                                path = os.path.join(self.data_dirs_ssn, path.strip('/'))
                                fn = os.path.splitext(os.path.basename(path))[0]
                                mask_path = os.path.join(self.mask_dirs_ssn, fn+'_mask.nii.gz')
                        else:
                            path = None
                            mask_path = None
                            
                        data['path%d'%i] = path
                        data['mask_path%d'%i] = mask_path
        
                    if (data['path1'] is not None) and ((data['path0'] is not None) or (data['path2'] is not None)):
                        self.path_list.append(data)
        
        if phase == 'train':
            self.transform = transforms.Compose([ 
                transforms.CenterPadCrop(self.crop_size+8, pad_value=-1024, dims=3),
                transforms.RandomPadCrop(self.crop_size, pad_value=-1024, dims=3),
                # transforms.CenterPadCrop(cf.crop_size, pad_value=-1024),
                # transforms.RandomFlipAxis(axis=1,coord_start_axis=1), #z
                transforms.RandomFlipAxis(axis=2,coord_start_axis=1), #y
                transforms.RandomFlipAxis(axis=3,coord_start_axis=1), #x
                # transforms.Normalize(to_zero_mean=True, vmin=-1000, vmax=600),
                # transforms.ZeroOut(4),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterPadCrop(cf.crop_size, pad_value=-1024),
                # transforms.Normalize(to_zero_mean=True, vmin=-1000, vmax=600),
            ])

    def __len__(self):
        return len(self.path_list)
    
    def _transform(self, image_zyx, mask_zyx, transform, transforms_more=None):
        inputs = {'data': image_zyx, 'mask':mask_zyx}
        inputs = transform(**inputs)
        image_zyx, mask_zyx = inputs['data'].copy(), inputs['mask'].copy()
        # normalize
        vmin, vmax = -1024, 1000  # 200,400
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
        # tt = int(round(ddays/30.0)) # normalize day, year
        tt = ddays/30.0          # normalize day, month
        assert tt>0, 'dd must greater than 0'

        gt_gimage, mask_zyx, ori_shape, pre_crop_size, sp_scale = self.load_image_transform(path0, path1, mask_path0, mask_path1)
        image_zyx0, image_zyx1 = gt_gimage
        mask_zyx0, mask_zyx1 = mask_zyx

        target_image, target_mask = image_zyx1[None], mask_zyx1[None]

        if self.phase=='train' and np.random.rand()<0.2:
            tt = 0
            target_image, target_mask = image_zyx0[None], mask_zyx0[None]

        vk, vb = self.rate_df[self.rate_df['nid']==fn]['vk'].values[0], self.rate_df[self.rate_df['nid']==fn]['vb'].values[0]
        mk, mb = self.rate_df[self.rate_df['nid']==fn]['mk'].values[0], self.rate_df[self.rate_df['nid']==fn]['mb'].values[0]
        lnv = vk*(1-math.exp(-vb*tt))
        lnm = mk*(1-math.exp(-mb*tt))

        if self.phase=='train':
            lnm = lnm + np.random.normal(0, 0.05)
            lnv = lnv + np.random.normal(0, 0.05)
        
        return {
            'inputs': [image_zyx0[None], mask_zyx0[None], tt], 
            'targets': [target_image, target_mask, lnv, lnm], 
            'uid': data['uid'],
            'label': data['label'],
            'transform': [np.array(ori_shape), pre_crop_size, sp_scale]
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
        pre_crop_size =  self.input_size/np.array(spacing_zyx)
        assert pre_crop_size[0]<=ori_shape[0] and pre_crop_size[1]<=ori_shape[1] and pre_crop_size[2]<=ori_shape[2], 'Pre-crop size must be smaller than ori-shape' 
        crop_func = transforms.CenterPadCrop(pre_crop_size, pad_value=-1024, dims=3)
        inputs = crop_func(data=image_zyx, mask=mask_zyx, centerd=None)
        image_zyx, mask_zyx = inputs['data'], inputs['mask']

        sp_scale = np.array(spacing_zyx)/self.resolution
        sp_scale = np.insert(sp_scale, 0, 1.0)
        image_zyx = zoom(image_zyx, sp_scale, order=1).astype(np.float32)
        mask_zyx = zoom(mask_zyx, sp_scale, order=0).astype(np.float32)

        image_zyx, mask_zyx = self._transform(image_zyx, mask_zyx, self.transform)

        return image_zyx, mask_zyx, ori_shape, pre_crop_size, sp_scale
