import cv2
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import math
from scipy.ndimage import zoom
import torch
import torch.nn as nn
from skimage.measure import compare_psnr, compare_ssim

from models.losses.dice_loss import Dice
from models.losses.dice_loss import ssim as ssim_func


def find_volume(mask, spacing_zyx=np.array([1,1,1])):
    non_slice_area = []
    mask = np.uint8(mask)

    for j in range(mask.shape[0]):
        mask_slice = mask[j, :, :]
        contours,hierarchy = cv2.findContours(mask_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
        area = []
        for m in range(len(contours)):      
            area.append(cv2.contourArea(contours[m]))
        if len(area) > 0:
            max_idx = np.argmax(area)
            max_area = cv2.contourArea(contours[max_idx])
            non_slice_area.append(max_area)

    area_sum = np.sum(non_slice_area)   
    volume = area_sum * spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]
    return volume


def find_info(mask, img, spacing_zyx=np.array([1,1,1])):
    mask = np.uint8(mask)

    each_slice_area = []  # 记录每张切片的面积
    each_slice_contours = []
    non_slice_area = []

    for j in range(mask.shape[0]):  # 遍历筛选面积最大的切片
        mask_slice = mask[j, :, :]
        contours,hierarchy = cv2.findContours(mask_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)        
        area = []

        for m in range(len(contours)):
            area.append(cv2.contourArea(contours[m]))
        if len(area) > 0:
            max_idx = np.argmax(area)
            max_area = cv2.contourArea(contours[max_idx])
            each_slice_area.append(max_area)
            non_slice_area.append(max_area)  # 记录所有非零的区域面积，用于球体的计算
            each_slice_contours.append(contours[max_idx])
        else:
            each_slice_area.append(0)
            each_slice_contours.append(0)
    max_area = np.max(each_slice_area)
    max_index = each_slice_area.index(max_area)

    if type(each_slice_contours[max_index]) is np.ndarray: 
        rec = cv2.minAreaRect(each_slice_contours[max_index])
    else:
        rec  = ((0, 0), (0, 0), 0)                    ### 防止找不到最大area

    if rec[1][0]>rec[1][1]:                           ### 计算最大外接矩形的最长边和最短边的长和宽
        Max_Diameter = rec[1][0]
        Min_Diameter = rec[1][1]
    else:
        Max_Diameter = rec[1][1]
        Min_Diameter = rec[1][0]
    Min_Diameter = Min_Diameter*spacing_zyx[1]
    Max_Diameter = Max_Diameter*spacing_zyx[1]

    #  ------------------------ 计算结节的体积（世界坐标）  ------------------------
    area_sum = np.sum(non_slice_area)
    volume = area_sum * spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]

    ### ---------------- 计算3D上结节的Hu密度 ------------------------
    solid = 0
    try:
        temp_list = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if mask[i][j][k] == 1:
                        hu = img[i][j][k]
                        if hu > -250:
                            solid += 1
                        if hu > 300:   # 若HU值大于600，则该像素点HU值取600，（即截段到600）
                            hu = 300 
                        hu = hu + 1000  # 截段后，所有HU值加上1000
                        temp_list.append(hu)
        if len(temp_list) != 0:
            mean_hu = np.mean(temp_list)    ### 平均HU值
            std_hu = np.std(temp_list)      ### HU值标准差
        else:
            mean_hu = 0
            std_hu = 0
    except Exception as e:
        print("Something wrong in getting nodule mean and std HU!...")

    ### ---------------- 计算3D上结节的质量 ------------------------
    mass = mean_hu * volume       

    return Max_Diameter, volume, mass


def eval_PSNR(img0, img1, mse):
    img0 = img0[0,0,:,:,:].cpu().numpy()
    img1 = img1[0,0,:,:,:].cpu().numpy()
    img0 = img0*255.
    img1 = img1*255.

    # diff = img0 - img1
    # mse = np.mean(np.square(diff))
    # maxvalue = 1
    # psnr = 10*np.log10(maxvalue*maxvalue/mse)

    psnr = compare_psnr(img0, img1, data_range=255)

    return psnr


# def eval_SSIM(img0, img1):
#     img0 = img0[0,0,:,:,:].cpu().numpy()
#     img1 = img1[0,0,:,:,:].cpu().numpy()
#     img0 = img0*255.
#     img1 = img1*255.

#     ssim = compare_ssim(img0, img1, data_range=255, multichannel=False)

#     return ssim


def eval_cls(y, pred):
    acc = metrics.accuracy_score(y, pred)

    fpr, tpr, thresholds = roc_curve(y, pred)
    auc_score = auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sen = np.nan if (tp+fn) == 0 else float(tp)/(tp+fn)
    spe = np.nan if (tn+fp) == 0 else float(tn)/(tn+fp)
    recall = np.nan if (tp+fn) == 0 else float(tp)/(tp+fn)
    gmean = math.sqrt(recall*spe)

    return acc, auc_score, sen, spe, gmean, tn, fp, fn, tp


def eval_PD(pred_img, pred_mask, input_img, input_mask, transform, threshold=0.1):
    ori_shape, pre_crop_size, sp_scale = transform
    ori_shape = (ori_shape.cpu().numpy()[0]).astype(int)
    pre_crop_size = pre_crop_size.cpu().numpy()[0]
    sp_scale = sp_scale.cpu().numpy()[0]

    pred_mask = pred_mask[0,0,:,:,:].cpu().numpy()
    input_mask = input_mask[0,0,:,:,:].cpu().numpy()
    pred_img = pred_img[0,0,:,:,:].cpu().numpy()
    input_img = input_img[0,0,:,:,:].cpu().numpy()

    pred_mask_copy = pred_mask.copy()
    pred_mask_copy[pred_mask_copy>=threshold]=1
    pred_mask_copy[pred_mask_copy<threshold]=0

    # 从inputsize 周围补0 到 pad_shape   (64->72)
    pad_shape = (pre_crop_size*sp_scale[1:]).astype(int)
    pred_mask_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-pred_mask_copy.shape[0])/2)
    d_y = np.int8((pad_shape[1]-pred_mask_copy.shape[1])/2)
    d_x = np.int8((pad_shape[2]-pred_mask_copy.shape[2])/2)
    pred_mask_zeros[d_z:d_z+pred_mask_copy.shape[0], d_y:d_y+pred_mask_copy.shape[1], d_x:d_x+pred_mask_copy.shape[2]] = pred_mask_copy

    input_mask_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-input_mask.shape[0])/2)
    d_y = np.int8((pad_shape[1]-input_mask.shape[1])/2)
    d_x = np.int8((pad_shape[2]-input_mask.shape[2])/2)
    input_mask_zeros[d_z:d_z+input_mask.shape[0], d_y:d_y+input_mask.shape[1], d_x:d_x+input_mask.shape[2]] = input_mask

    pred_img_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-pred_img.shape[0])/2)
    d_y = np.int8((pad_shape[1]-pred_img.shape[1])/2)
    d_x = np.int8((pad_shape[2]-pred_img.shape[2])/2)
    pred_img_zeros[d_z:d_z+pred_img.shape[0], d_y:d_y+pred_img.shape[1], d_x:d_x+pred_img.shape[2]] = pred_img

    input_img_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-input_img.shape[0])/2)
    d_y = np.int8((pad_shape[1]-input_img.shape[1])/2)
    d_x = np.int8((pad_shape[2]-input_img.shape[2])/2)
    input_img_zeros[d_z:d_z+input_img.shape[0], d_y:d_y+input_img.shape[1], d_x:d_x+input_img.shape[2]] = input_img

    # zoom回pre_crop_size (72->72)
    shape_scale = pre_crop_size/pad_shape
    pred_mask_zeros = zoom(pred_mask_zeros, shape_scale, order=0)
    input_mask_zeros = zoom(input_mask_zeros, shape_scale, order=0)
    pred_img_zeros = zoom(pred_img_zeros, shape_scale, order=1)
    input_img_zeros = zoom(input_img_zeros, shape_scale, order=1)

    # 周围补0到原始图像大小 (72->100)
    pred_mask_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-pred_mask_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-pred_mask_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-pred_mask_zeros.shape[2])/2)
    pred_mask_ori[d_z:d_z+pred_mask_zeros.shape[0], d_y:d_y+pred_mask_zeros.shape[1], d_x:d_x+pred_mask_zeros.shape[2]] = pred_mask_zeros

    input_mask_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-input_mask_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-input_mask_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-input_mask_zeros.shape[2])/2)
    input_mask_ori[d_z:d_z+input_mask_zeros.shape[0], d_y:d_y+input_mask_zeros.shape[1], d_x:d_x+input_mask_zeros.shape[2]] = input_mask_zeros

    pred_img_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-pred_img_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-pred_img_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-pred_img_zeros.shape[2])/2)
    pred_img_ori[d_z:d_z+pred_img_zeros.shape[0], d_y:d_y+pred_img_zeros.shape[1], d_x:d_x+pred_img_zeros.shape[2]] = pred_img_zeros

    input_img_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-input_img_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-input_img_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-input_img_zeros.shape[2])/2)
    input_img_ori[d_z:d_z+input_img_zeros.shape[0], d_y:d_y+input_img_zeros.shape[1], d_x:d_x+input_img_zeros.shape[2]] = input_img_zeros

    d0, v0, m0 = find_info(input_mask_ori, input_img_ori)
    d1, v1, m1 = find_info(pred_mask_ori, pred_img_ori)

    if (d1-d0)>1.5:
        label_d = 1
    else:
        label_d = 0

    if (m1/m0)>1.25:
        label_m = 1
    else:
        label_m = 0

    return label_d, label_m


def eval_Dice(pred_mask, target_mask, transform):
    ori_shape, pre_crop_size, sp_scale = transform
    ori_shape = (ori_shape.cpu().numpy()[0]).astype(int)
    pre_crop_size = pre_crop_size.cpu().numpy()[0]
    sp_scale = sp_scale.cpu().numpy()[0]

    # 从inputsize 周围补0 到 pad_shape   (64->72)
    pad_shape = (pre_crop_size*sp_scale[1:]).astype(int)
    pred_mask_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-pred_mask.shape[0])/2)
    d_y = np.int8((pad_shape[1]-pred_mask.shape[1])/2)
    d_x = np.int8((pad_shape[2]-pred_mask.shape[2])/2)
    pred_mask_zeros[d_z:d_z+pred_mask.shape[0], d_y:d_y+pred_mask.shape[1], d_x:d_x+pred_mask.shape[2]] = pred_mask

    target_mask_zeros = np.zeros(pad_shape)
    d_z = np.int8((pad_shape[0]-target_mask.shape[0])/2)
    d_y = np.int8((pad_shape[1]-target_mask.shape[1])/2)
    d_x = np.int8((pad_shape[2]-target_mask.shape[2])/2)
    target_mask_zeros[d_z:d_z+target_mask.shape[0], d_y:d_y+target_mask.shape[1], d_x:d_x+target_mask.shape[2]] = target_mask

    # zoom回pre_crop_size (72->72)
    shape_scale = pre_crop_size/pad_shape
    pred_mask_zeros = zoom(pred_mask_zeros, shape_scale, order=1)
    target_mask_zeros = zoom(target_mask_zeros, shape_scale, order=0)

    # 周围补0到原始图像大小 (72->100)
    pred_mask_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-pred_mask_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-pred_mask_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-pred_mask_zeros.shape[2])/2)
    pred_mask_ori[d_z:d_z+pred_mask_zeros.shape[0], d_y:d_y+pred_mask_zeros.shape[1], d_x:d_x+pred_mask_zeros.shape[2]] = pred_mask_zeros

    target_mask_ori = np.zeros(ori_shape)
    d_z = np.int8((ori_shape[0]-target_mask_zeros.shape[0])/2)
    d_y = np.int8((ori_shape[1]-target_mask_zeros.shape[1])/2)
    d_x = np.int8((ori_shape[2]-target_mask_zeros.shape[2])/2)
    target_mask_ori[d_z:d_z+target_mask_zeros.shape[0], d_y:d_y+target_mask_zeros.shape[1], d_x:d_x+target_mask_zeros.shape[2]] = target_mask_zeros

    dice_func = Dice().loss
    dice = 1-dice_func(torch.from_numpy(pred_mask_ori), torch.from_numpy(target_mask_ori))

    return dice


def eval_reg(lnv_tgt, lnm_tgt, lnv_pred, lnm_pred, label):
    lnv_tgt = [np.exp(x) for x in lnv_tgt]
    lnm_tgt = [np.exp(x) for x in lnm_tgt]
    lnv_pred = [np.exp(x) for x in lnv_pred]
    lnm_pred = [np.exp(x) for x in lnm_pred]

    lnv_mse = mean_squared_error(lnv_tgt, lnv_pred)
    lnv_mae = mean_absolute_error(lnv_tgt, lnv_pred)
    lnv_r2 = r2_score(lnv_tgt, lnv_pred)

    lnm_mse = mean_squared_error(lnm_tgt, lnm_pred)
    lnm_mae = mean_absolute_error(lnm_tgt, lnm_pred)
    lnm_r2 = r2_score(lnm_tgt, lnm_pred)

    # lnm_pred_label = np.int64(lnm_pred > np.log(1.25))
    # acc = metrics.accuracy_score(label, lnm_pred_label)
    # tn, fp, fn, tp = confusion_matrix(label, lnm_pred_label).ravel()

    return lnv_mse, lnv_mae, lnv_r2, lnm_mse, lnm_mae, lnm_r2



def evaluation(pred, target_img, target_mask, input_img, input_mask, transform=None):
    pred_img, pred_mask = pred[:,:1,:,:,:], pred[:,1:,:,:,:]

    dice_func = Dice().loss
    mse_func = nn.MSELoss()

    dice = 1-dice_func(pred_mask, target_mask)

    mse = mse_func(pred_img, target_img)
    mse_nod = mse_func(pred_img*target_mask, target_img*target_mask)
    if torch.sum(target_mask)!=0:
        _,_,z,y,x = torch.where(target_mask)
        z,y,x = z.cpu().numpy(), y.cpu().numpy(), x.cpu().numpy()
        z_min,z_max,y_min,y_max,x_min,x_max = max(np.min(z)-5,0), min(np.max(z)+5,target_mask.shape[2]), max(np.min(y)-5, 0), min(np.max(y)+5,target_mask.shape[3]), max(np.min(x)-5,0), min(np.max(x)+5,target_mask.shape[4])
        mse_nod2 = mse_func(pred_img[:,:,z_min:z_max,y_min:y_max,x_min:x_max]*target_mask[:,:,z_min:z_max,y_min:y_max,x_min:x_max], target_img[:,:,z_min:z_max,y_min:y_max,x_min:x_max]*target_mask[:,:,z_min:z_max,y_min:y_max,x_min:x_max])
    else:
        mse_nod2 = 0
    
    
    psnr = eval_PSNR(target_img, pred_img, mse.cpu().numpy())
    # ssim = eval_SSIM(target_img, pred_img)
    ssim = ssim_func(target_img, pred_img)
    ssim = ssim.cpu().numpy()
    
    return dice, mse, mse_nod, mse_nod2, psnr, ssim