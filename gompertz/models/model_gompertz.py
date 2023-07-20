# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from timm.models.layers import trunc_normal_
from models.vit_3d import VisionTransformerND, ConvStem3D


class LNGrowthNet(nn.Module):
    def __init__(self, args):

        super(LNGrowthNet, self).__init__()
        self.setup_seed(45)
        self.backbone = VisionTransformerND(
            stop_grad_conv1=False, is_3d=True, use_learnable_pos_emb=False,
            img_size=args.input_size, in_chans=2,
            patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, 
            num_classes=args.nb_classes, drop_rate=args.drop, drop_path_rate=args.drop_path)

        self.num_features = self.backbone.embed_dim
        self.prefeat_token = nn.Parameter(torch.zeros(1, self.num_features))
        trunc_normal_(self.prefeat_token, std=.02)
        self.vk = nn.parameter.Parameter(torch.tensor(2.0), requires_grad=True)
        self.mk = nn.parameter.Parameter(torch.tensor(2.0), requires_grad=True)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        self.mlp_k = nn.Sequential(
            nn.Linear(self.num_features*2, self.num_features, bias=False),
            # nn.Linear(self.num_features, self.num_features, bias=False),   # withoutZ
            norm_layer(self.num_features),
            act_layer(),
            nn.Linear(self.num_features, 2, bias=True),
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(self.num_features, self.num_features, bias=False),
            norm_layer(self.num_features),
            act_layer(),
            nn.Linear(self.num_features, 2, bias=True),
        )
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, inputs, targets, phase='train'):
        image_zyx0, mask_zyx0, v0, m0, v1, m1, tt = inputs
        target_image, target_mask, lnv, lnm, label = targets

        lnv, lnm = lnv.float(), lnm.float()
        v0, m0, tt = v0.float(), m0.float(), tt.float()

        fg, fl = self.backbone(torch.cat([image_zyx0,mask_zyx0],dim=1))
        prefeat_tokens = self.prefeat_token.expand(fg.size(0), -1)
        vmk = self.mlp_k(torch.cat([fg, prefeat_tokens], dim=1))
        # vmk = self.mlp_k(fg)      # withoutZ
        
        vk, mk = vmk[:,0], vmk[:,1]
        vmb = self.mlp_b(fl)
        vb, mb = vmb[:,0], vmb[:,1]

        predlnv = vk*(1-torch.exp(-vb*tt))
        predlnm = mk*(1-torch.exp(-mb*tt))
    
        loss_vrate = self.smooth_l1(predlnv, lnv)
        loss_mrate = self.smooth_l1(predlnm, lnm)

        loss = loss_vrate + loss_mrate

        result = {'loss': loss, 'loss_vrate': loss_vrate, 'loss_mrate': loss_mrate, 
                  'predlnv':predlnv, 'predlnm':predlnm
                }

        return result, {}
