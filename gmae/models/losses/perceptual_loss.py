import torch
from torch import nn
from models.probunet.resnet3d import generate_model


def vgg_loss(custom_vgg, target, pred, weights=None, vgg_feat_weights=None):
    """
    :param custom_vgg:
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)
    if weights is None:

        loss = torch.cat(
            [
                vgg_feat_weights[i]
                * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3, 4]).unsqueeze(dim=-1)
                for i, (tf, pf) in enumerate(zip(target_feats, pred_feats))
            ],
            dim=-1,
        )
    else:
        pix_loss = [
            vgg_feat_weights[0]
            * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
            .unsqueeze(dim=-1)
            .to(torch.float)
        ]
        loss = torch.cat(
            pix_loss
            + [
                vgg_feat_weights[i + 1]
                * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3, 4]).unsqueeze(dim=-1)
                for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:]))
            ],
            dim=-1,
        )

    loss = torch.sum(loss, dim=1).mean()
    return loss


class Encoder(nn.Module):

    def __init__(self, dim=64, mlp=True, checkpoint_path=None):
        super().__init__()

        self.encoder = generate_model(model_depth=34, planes=[32,64,128,256], n_input_channels=1, n_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]
        if mlp: 
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)       
        
        self.mlp_text = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp, bias=False),
            nn.LayerNorm(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, 3, bias=True)
        )
        self.mlp_mal = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp, bias=False),
            nn.LayerNorm(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, 5, bias=True)
        )
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            pretrain_dict = checkpoint['state_dict']
            pretrain_dict = {k.split('encoder_k.')[1]:v for k,v in pretrain_dict.items() if 'encoder_k' in k}
            print('*** start load parameters***')
            model_dict = self.state_dict()
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}     
            model_dict.update(pretrain_dict)
            self.load_state_dict(model_dict)
            print('load params: {}/{}'.format(len(pretrain_dict), len(model_dict)))

    def forward(self, x):
        feats = []
        x = self.encoder.conv1(x)
        x = self.encoder.layer1(x)
        feats.append(x)
        x = self.encoder.layer2(x)
        feats.append(x)
        x = self.encoder.layer3(x)
        feats.append(x)
        x = self.encoder.layer4(x)
        feats.append(x)

        return feats


class PerceptualLossInstances(torch.nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path = 'models/probunet/med_moco_models/model_089.tar'
        self.custom_vgg = Encoder(checkpoint_path=checkpoint_path)
        # for param in self.custom_vgg.parameters():
        #     param.requires_grad = False
        self.custom_vgg.eval()

        self.vgg_feat_weights = [1.0, 1.0, 1.0, 1.0]

    def forward(self, image, target):
        loss = vgg_loss(self.custom_vgg, target, image, vgg_feat_weights=self.vgg_feat_weights)
        return loss