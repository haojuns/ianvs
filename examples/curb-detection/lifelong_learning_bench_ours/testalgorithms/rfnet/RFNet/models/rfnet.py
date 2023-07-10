import torch.nn as nn
import torch
from itertools import chain # 串联多个迭代对象
import numpy as np
from .util import _BNReluConv, upsample

import pdb

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
    def __init__(self, backbone, num_classes, use_bn=True):
        super(RFNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        print(self.backbone.num_features)
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, rgb_inputs, depth_inputs = None, embedding = False):
        #pdb.set_trace()
        x, additional, embedding_feats = self.backbone(rgb_inputs, depth_inputs)
        logits = self.logits.forward(x)
        logits_upsample = upsample(logits, rgb_inputs.shape[2:])
        #print(logits_upsample.size)

        if embedding:
            # 计算style_code
            embedding_code = self.calc_style_std_mean(torch.unsqueeze(embedding_feats[0][0, :, :, :], 0))
            embedding_code = np.append(embedding_code, self.calc_style_std_mean(torch.unsqueeze(embedding_feats[1][0, :, :, :], 0)))
            
            return logits_upsample, embedding_code
        else:
            return logits_upsample

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