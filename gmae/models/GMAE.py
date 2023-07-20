import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_

from models.losses.dice_loss import SoftDiceLoss, DC_and_CE_loss, Dice, SSIM
from models.losses.perceptual_loss import PerceptualLossInstances
from evaluation import evaluation

class ResBlock(torch.nn.Module):

    expansion = 1

    def __init__(self, channels, strides, num_groups=None):
        super(ResBlock, self).__init__()
        if num_groups is None:
            num_groups = min(channels[1]//4, 32)
        self.conv1 = nn.Conv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=strides[0], padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=channels[1], out_channels=channels[1], kernel_size=3, stride=strides[1], padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels[1])
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels[1])
        self.downsample = None
        if channels[0] != channels[1] or strides[0] !=1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=1, stride=strides[0], bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=channels[1])
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def match_to(x, ref, keep_axes=(1,)):

    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        pass
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        while x.dim() < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x


class Encoder(nn.Module):

    def __init__(self, in_channels, filters, n_blocks, strides, injection_channels=0, injection_depth=None, basic_module=ResBlock):
        super(Encoder, self).__init__()

        self.inplanes = filters[0]
        self.injection_channels = injection_channels
        self.injection_depth = injection_depth
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=self.inplanes//4, num_channels=self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            if injection_depth is not None:
                if (injection_depth == -1) or (i ==injection_depth):
                    self.inplanes = self.inplanes + self.injection_channels
            layer = self._make_layer(basic_module, filters[i], n_blocks[i], strides[i])
            self.layers.append(layer)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(channels=[self.inplanes, planes], strides=[stride,1]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(channels=[self.inplanes, planes], strides=[1,1]))
        return nn.Sequential(*layers)

    def forward(self, x, injection=None):
        x = self.conv1(x)
        encoder_feats = []
        for i in range(len(self.layers)):
            if ((self.injection_depth == -1) or (i ==self.injection_depth)) and (injection is not None) and (self.injection_channels>0):
                if isinstance(injection, (list, tuple)):
                    assert injection[i].size(1) == self.injection_channels, 'channels of injection do not equal injection_channels'
                    injection_x = match_to(injection[i], x, (0, 1))
                    x = torch.cat((x, injection_x), 1)
                else:
                    assert injection.size(1) == self.injection_channels, 'channels of injection do not equal injection_channels'
                    injection_x = match_to(injection, x, (0, 1))
                    x = torch.cat((x, injection_x), 1)
            x = self.layers[i](x)
            encoder_feats.append(x)     
        # top to bottom
        return encoder_feats[::-1]


class Decoder(nn.Module):

    def __init__(self, in_channels, filters, n_blocks, upstrides, injection_channels=0, injection_depth=None, basic_module=ResBlock):
        super(Decoder, self).__init__()

        self.inplanes = in_channels
        self.injection_channels = injection_channels
        self.injection_depth = injection_depth
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # lateral modules
        self.lateral_modules = nn.ModuleList()
        # posthoc modules
        self.posthoc_modules = nn.ModuleList()
        # up
        self.convups = nn.ModuleList()
        for i in range(len(filters)):
            self.lateral_modules.append(nn.Conv3d(filters[i], filters[i], 1, 1, 0, bias=False))
            if upstrides[i]==2:
                self.convups.append(nn.Sequential(
                    nn.Conv3d(self.inplanes, filters[i], 1, 1, 0, bias=False),
                    self.up
                ))
            else:
                self.convups.append(nn.Conv3d(self.inplanes, filters[i], 1, 1, 0, bias=False))
            self.inplanes = filters[i]
            if injection_depth is not None:
                if (injection_depth == -1) or (i ==injection_depth):
                    self.inplanes = self.inplanes + self.injection_channels
            layer = self._make_layer(basic_module, filters[i], n_blocks[i], 1)
            self.posthoc_modules.append(layer)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(channels=[self.inplanes, planes], strides=[stride,1]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(channels=[self.inplanes, planes], strides=[1,1]))
        return nn.Sequential(*layers)

    def forward(self, x, encoder_feats, injection=None):
        """
        encoder_feats: top to bottom
        """
        for i in range(len(self.lateral_modules)):
            x = self.convups[i](x)
            lat = self.lateral_modules[i](encoder_feats[i])
            x = x+lat
            if ((self.injection_depth == -1) or (i ==self.injection_depth)) and (injection is not None) and (self.injection_channels>0):
                if isinstance(injection, (list, tuple)):
                    assert injection[i].size(1) == self.injection_channels, 'channels of injection do not equal injection_channels'
                    injection_x = match_to(injection[i], x, (0, 1))
                    x = torch.cat((x, injection_x), 1)
                else:
                    assert injection.size(1) == self.injection_channels, 'channels of injection do not equal injection_channels'
                    injection_x = match_to(injection, x, (0, 1))
                    x = torch.cat((x, injection_x), 1)
            x = self.posthoc_modules[i](x)
        return x


class SimpleDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SimpleDown, self).__init__()

        stem = []
        input_dim, output_dim = in_channels, 32
        conv_length = int(np.log2(out_channels/output_dim))+1
        for l in range(conv_length):
            stem.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            # stem.append(nn.BatchNorm3d(output_dim))
            stem.append(nn.GroupNorm(num_groups=output_dim//4, num_channels=output_dim))
            stem.append(nn.ReLU(inplace=True))
            # stem.append(nn.GroupNorm(num_groups=32, num_channels=output_dim))
            # stem.append(nn.GELU())
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv3d(input_dim, out_channels, kernel_size=1))
        self.proj = nn.Sequential(*stem)

    def forward(self, x):
        x = self.proj(x)    
        return x


class NGrowth(nn.Module):
  
    def __init__(self, in_channels, out_channels, time_channels=8, growth_channels=8, injection_depth=-1):
        super(NGrowth, self).__init__()

        # create encoder path
        self.num_filters = [32, 64, 128, 128]  
        n_blocks = [1,2,2,2]
        strides = [1,2,2,2]

        self.endocers = Encoder(in_channels, self.num_filters, n_blocks, strides)

        self.shape_decoders = Decoder(self.num_filters[-1], self.num_filters[::-1], n_blocks=[2,2,2,2], upstrides=[1, 2, 2, 2],
                                injection_channels=growth_channels+time_channels, injection_depth=injection_depth)
        self.text_decoders = Decoder(self.num_filters[-1], self.num_filters[::-1], n_blocks=[2,2,2,2], upstrides=[1, 2, 2, 2],
                                injection_channels=growth_channels+time_channels, injection_depth=injection_depth)

        self.downshape = SimpleDown(out_channels, self.num_filters[-1])
        self.downtext = SimpleDown(out_channels*2, self.num_filters[-1])
        # self.downtext = SimpleDown(out_channels, self.num_filters[-1])
     
        self.conv_shape = nn.Sequential(
            nn.Conv3d(self.num_filters[0], self.num_filters[0], 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=self.num_filters[0]//4, num_channels=self.num_filters[0]),
            nn.LeakyReLU(),
            nn.Conv3d(self.num_filters[0], out_channels, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        self.conv_text = nn.Sequential(
            nn.Conv3d(self.num_filters[0], self.num_filters[0], 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=self.num_filters[0]//4, num_channels=self.num_filters[0]),
            nn.LeakyReLU(),
            nn.Conv3d(self.num_filters[0], out_channels, 1, 1, 0, bias=True),
            # nn.Sigmoid()
            nn.Tanh()
        )
        self.conv_shape[-2].bias.data.fill_(-1.0)

    def forward(self, encoder_inputs, text_inputs, shape_inputs, source_time=None, target_time=None):
        
        encoders_features = self.endocers(encoder_inputs)
        lnv_embedding, lnm_embedding = target_time
        # shape
        x_shape = self.downshape(shape_inputs)
        x_shape = self.shape_decoders(x_shape, encoders_features, injection=torch.exp(lnv_embedding))
        x_shape = self.conv_shape(x_shape)
        # texture
        text_shape_inputs = torch.cat([text_inputs, x_shape], dim=1)
        x_texture = self.downtext(text_shape_inputs)
        x_texture = self.text_decoders(x_texture, encoders_features, injection=torch.exp(lnm_embedding))
        x_texture = self.conv_text(x_texture)
        x_texture = x_texture*x_shape + text_inputs

        x = torch.cat([x_texture, x_shape], dim=1)
        return x


class GrowthF(nn.Module):
  
    def __init__(self, in_channels):
        super(GrowthF, self).__init__()

        # create encoder path
        self.num_filters = [32, 64, 128, 256]  
        n_blocks = [1,2,2,2]
        strides = [1,2,2,2]

        self.endocers = Encoder(in_channels, self.num_filters, n_blocks, strides)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.prefeat_token = nn.Parameter(torch.zeros(1,  self.num_filters[-1]))
        trunc_normal_(self.prefeat_token, std=.02)
        self.mlp_k = nn.Sequential(
            nn.Linear(self.num_filters[-1]*2, self.num_filters[-1], bias=False),
            nn.LayerNorm(self.num_filters[-1]),
            nn.LeakyReLU(),
            nn.Linear(self.num_filters[-1], 2, bias=True),
            # nn.ReLU()
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(self.num_filters[-1], self.num_filters[-1], bias=False),
            nn.LayerNorm(self.num_filters[-1]),
            nn.LeakyReLU(),
            nn.Linear(self.num_filters[-1], 2, bias=True),
            # nn.ReLU()
        )
    
    def forward(self, encoder_inputs, t):
        
        encoders_features = self.endocers(encoder_inputs)
        feat = self.pool(encoders_features[0])
        feat = feat.view(feat.size(0), -1)
        prefeat_tokens = self.prefeat_token.expand(feat.size(0), -1)
        vmk = self.mlp_k(torch.cat([feat, prefeat_tokens], dim=1))
        vk, mk = vmk[:,0], vmk[:,1]
        vmb = self.mlp_b(feat)
        vb, mb = vmb[:,0], vmb[:,1]

        lnv = vk*(1-torch.exp(-vb*t))
        lnm = mk*(1-torch.exp(-mb*t))

        return lnv, lnm, torch.exp(vk), torch.exp(mk)


class Net(nn.Module):
    def __init__(self, cf=None, logger=None):
        super().__init__()
        # define required parameters
        time_channels=1
        #d_injection_depth=-1, all
        self.ngrowth = NGrowth(in_channels=1, out_channels=1, time_channels=time_channels, growth_channels=0, 
                                injection_depth=-1)
        
        # ##### if noGompertz #####
        # self.ngrowth = NGrowth(in_channels=1, out_channels=1, time_channels=time_channels, growth_channels=0, 
        #                         injection_depth=None)

        self.fgrowth = GrowthF(in_channels=1)

        self.mse = nn.MSELoss(reduction='sum')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')
        self.dice_loss = SoftDiceLoss()
        self.dice_loss_func = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': True}, {})
        self.perceptual_loss = PerceptualLossInstances()
        self.dice_eval = Dice().loss
        self.ssim_loss = SSIM()

    def forward(self, inputs, targets, transform=None, phase='train', epoch=0):
        image_zyx0, mask_zyx0, tt = inputs
        target_image, target_mask, lnv, lnm = targets

        lnv, lnm = lnv.float(), lnm.float()
        tt = tt.float()
    
        # growth
        encoder_inputs = image_zyx0
        text_inputs = image_zyx0
        shape_inputs = mask_zyx0
        
        # predlnv, predlnm, VMax, MMax = self.fgrowth(encoder_inputs, tt)
        # if phase == 'train' and np.random.rand()<0.3:
        #     lnm_embedding = lnm.unsqueeze(1)
        #     lnv_embedding = lnv.unsqueeze(1)
        # else:
        #     lnm_embedding = predlnm.detach().unsqueeze(1)
        #     lnv_embedding = predlnv.detach().unsqueeze(1)

        lnm_embedding = lnm.unsqueeze(1)
        lnv_embedding = lnv.unsqueeze(1)

        pred = self.ngrowth(encoder_inputs, text_inputs, shape_inputs, target_time=[lnv_embedding, lnm_embedding])

        if phase=='infer': 
            dice, mse, mse_nod, mse_nod2, psnr, ssim = evaluation(pred, target_image, target_mask, image_zyx0, mask_zyx0, transform)
            return pred, dice, mse, mse_nod, mse_nod2, psnr, ssim

        mask_or = ((pred[:,1:2]>0.3) | (target_mask>0.3)).float()
        loss_reconbg = self.mse(pred[:,:1]*(1-mask_or), image_zyx0*(1-mask_or)) / max((1-mask_or).sum(),1.0)
        loss_reconfg = (torch.abs(pred[:,:1]*mask_or - target_image*mask_or).sum(dim=[1,2,3,4]) / mask_or.sum(dim=[1,2,3,4])).mean()
        pred_central = central_up(pred[:,:1], central_size=32, shape_size=64)
        tg_central = central_up(target_image, central_size=32, shape_size=64)
        loss_perceptual = self.perceptual_loss(tg_central, pred_central)
      
        # shape
        loss_dice = self.dice_loss_func(pred[:,1:2], target_mask)
        dice_ori = 1-self.dice_loss(mask_zyx0, target_mask)
        dice_pred = 1-self.dice_loss((pred[:,1:2]>0.5).float(), target_mask)

        loss = loss_dice + loss_reconbg + loss_perceptual + loss_reconfg 

        if loss>10:
            loss = loss*0.1

        result = {'loss': loss, 
            'loss_reconfg':loss_reconfg.detach(), 
            'loss_reconbg':loss_reconbg.detach(),
            'loss_perceptual':loss_perceptual.detach(),
            'loss_dice':loss_dice.detach(),
            'dice_ori':dice_ori.detach(),
            'dice_pred':dice_pred.detach(),
        }
        fmap = {
                'predmask': pred[:,1:2,32].detach(), 
                'pred': pred[:,:1,32].detach(), 
                'gt': target_image[:,:,32].detach(),
                'cur': image_zyx0[:,:,32].detach(),
        }

        return result, fmap


def central_up(image, central_size=32, shape_size=64):
    up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    diff = shape_size-central_size
    min_crop = diff//2
    max_crop = min_crop + central_size
    central_voi = image[:,:,min_crop:max_crop,min_crop:max_crop,min_crop:max_crop]
    central_voi = up(central_voi)
    return central_voi