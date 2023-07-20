import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp

from models.losses.ND_Crossentropy import CrossentropyND
from models.losses.tensor_utilities import sum_tensor


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1-dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.bce = nn.BCELoss()
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=None, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=None, **soft_dice_kwargs) #softmax_helper

    def forward(self, net_output, target):
        # net_output = F.sigmoid(net_output)
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        # ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        ce_loss = self.bce(net_output.reshape(-1), target.reshape(-1)) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result



def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class DiceLoss(torch.nn.Module):
    def __init__(self, dims=[2,3,4]):
        super(DiceLoss, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, is_softmax=False):
        pd, ph, pw = predict.size(2), predict.size(3), predict.size(4)
        d, h, w = gt.size(2), gt.size(3), gt.size(4)
        if ph != h or pw != w or pd != d:
            predict = F.upsample(input=predict, size=(d, h, w), mode='trilinear')

        predict = predict.float()
        gt = gt.float()
        if is_softmax:
            probability = F.softmax(predict, dim=1)
        else:
            probability = F.sigmoid(predict)

        intersection = torch.sum(probability*gt, dim=self.dims)
        union = torch.sum(probability*probability, dim=self.dims) + torch.sum(gt*gt, dim=self.dims)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice

        return dice_loss


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1-dice




class RegressLoss(nn.Module):
    def __init__(self, reduction='sum'):
        super().__init__()
        self.smoothl1_loss = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, inputs, target, label_weight):
        '''Calculate the smooth-ls loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        '''
        
        valid_indices = label_weight > 0
        device = inputs.device
        if valid_indices.sum():
            t = target[valid_indices]
            p = inputs[valid_indices]
            return self.smoothl1_loss(p, t)
        else:
            return torch.tensor(0).to(device).float()



class BCE_L1Class(nn.Module):
    def __init__(self, alpha=0.25):
        super(BCE_L1Class, self).__init__()
        self.alpha = alpha
        self.beta = 0.6
        self.l1_loss_func = nn.SmoothL1Loss()
        self.l1_loss_func_p = RegressLoss(reduction='sum')
        self.l1_loss_func_n = RegressLoss(reduction='sum')

    def forward(self, probas, labels):
        """
        probas: [B, C].
        labels: [B].
        """
        # label_c = labels>=self.alpha
       
        # loss_c = self.bce_loss_func(probas[:,0]-self.alpha, label_c.float())
        # loss_r = self.l1_loss_func(probas[:,0], labels)

        # loss = loss_c + loss_r*3.0

        pos_idcs = labels >= 1.1
        neg_idcs = labels < 1.1
        # loss_p = self.l1_loss_func_p(probas[:,0], labels, pos_idcs)
        # loss_n = self.l1_loss_func_n(probas[:,0], labels, neg_idcs)
        loss_p = self.l1_loss_func_p(probas, labels, pos_idcs)
        loss_n = self.l1_loss_func_n(probas, labels, neg_idcs)
        loss = loss_p*2.4 + loss_n

        num_pos = max(1, pos_idcs.sum())
        loss = loss / num_pos

        return loss


def maxmin_loss(pred, label):
    pos_idx = torch.where(label==1)
    neg_idx = torch.where(label==0)

    pred_pos = pred[pos_idx]
    pred_neg = pred[neg_idx]

    if len(pred_pos)==0:
        return torch.std(pred_neg), torch.std(pred_neg)
    elif len(pred_neg)==0:
        return torch.std(pred_pos), torch.std(pred_pos)

    std = torch.std(pred_pos)+torch.std(pred_neg)   
    mean = -torch.abs(torch.mean(pred_pos)-torch.mean(pred_neg))

    return std*10, mean*10



class BCELoss(nn.Module):
    
    def __init__(self, alpha=None, reduction='sum'):
        '''reduction: mean/sum'''
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, target):
        '''Calculate the focal loss.
        Args:
            inputs (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
        '''
        device = inputs.device
        t = target.squeeze().float()
        p = inputs.squeeze().float()
        w = 0.75 * t + (1 - 0.75) * (1 - t)
        # w = 0.9 * t + (1 - 0.9) * (1 - t)

        return F.binary_cross_entropy_with_logits(p, t, w)




#### SSIM loss ###########

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)