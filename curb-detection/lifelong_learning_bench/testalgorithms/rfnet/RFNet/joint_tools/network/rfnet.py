import torch.nn as nn
import torch
import numpy as np
from itertools import chain # 串联多个迭代对象

from .util import _BNReluConv, upsample
from network.resnet.resnet_single_scale_single_attention import *

#返回N张图片按channel为单位计算的std和mean
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    #计算（N,C,W*H)中第三维的var和mean
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class RFNet(nn.Module):
    def __init__(self, num_classes, backbone, criterion=None, criterion_aux=None, args=None):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.args = args
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=True)

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True, dsbn=False, mode=None, domain_label=None):
        x_size = x.size()
        x, _, style_feats = self.backbone(x, None)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, x_size[2:])

        # 计算style_code
        style_code = self.calc_style_std_mean(torch.unsqueeze(style_feats[0][0, :, :, :], 0))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[1][0, :, :, :], 0)))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[2][0, :, :, :], 0)))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[3][0, :, :, :], 0)))
        style_code = np.append(style_code, self.calc_style_std_mean(torch.unsqueeze(style_feats[4][0, :, :, :], 0)))

        # 当前是训练/推理阶段
        if self.training:
            loss1 = self.criterion(logits_upsample, gts)
            loss2 = loss1 # 没有设置aux_loss
            return_loss = [loss1, loss2]

            return return_loss
        else:
            return logits_upsample, style_code

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
    
    def calc_style_std_mean(self, target):
        input_mean, input_std = calc_mean_std(target)
        input_std=input_std.cpu()
        input_mean=input_mean.cpu()
        mean = input_mean.detach().numpy()
        std = input_std.detach().numpy()

        return np.append(mean, std)

def rfnet(args, num_classes, criterion, criterion_aux):
    """
    rfnet Network
    """
    resnet = resnet18(pretrained=True, efficient=False, use_bn=True) # 骨干网络选取的是ResNet-18

    print("Model : RFNet, Backbone : ResNet-18")
    return RFNet(num_classes, resnet, criterion=criterion, criterion_aux=criterion_aux, args=args)