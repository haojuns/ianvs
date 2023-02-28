import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import network.mynn as mynn
from network.dsbn import DomainSpecificBatchNorm2d

from torch.nn.modules.pooling import MaxPool2d as _MaxPool2d
from torch.nn.modules.activation import ReLU as _ReLU
from torch.nn.modules.conv import Conv2d as _Conv2d
from collections import OrderedDict
import operator
from itertools import islice

__all__ = ['VGG16_bn']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}

# 有Domain Label的Sequential
class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2

# 满足TwoInputSequential的基础模块重写
class Conv2d(_Conv2d):
    def forward(self, input, domain_label):
        return self._conv_forward(input, self.weight, self.bias), domain_label

class ReLU(_ReLU):
    def forward(self, input, domain_label):
        return F.relu(input, inplace=self.inplace), domain_label

class MaxPool2d(_MaxPool2d):
    def forward(self, input, domain_label):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, ceil_mode=self.ceil_mode,
                            return_indices=self.return_indices), domain_label

# VGG模块
def conv_layer(chann_in, chann_out, k_size, p_size, num_domains):
    layer = TwoInputSequential(
        Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        DomainSpecificBatchNorm2d(chann_out, num_domains),
        ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, num_domains=1):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i], num_domains) for i in range(len(in_list))]
    layers += [MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return TwoInputSequential(*layers)

def vgg_fc_layer(size_in, size_out, dropout):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )
    return layer

# VGG搭建 —— https://github.com/msyim/VGG16/blob/master/VGG16.py
class VGG16_bn(nn.Module):
    def __init__(self, dsbn=False, num_domains=None, num_classes=1000, dropout=0.5):
        self.inplanes = 64
        self.dsbn = dsbn
        self.num_domains = num_domains
        super(VGG16_bn, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2, num_domains=num_domains)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2, num_domains=num_domains)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2, num_domains=num_domains)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2, num_domains=num_domains)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2, num_domains=num_domains)

        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 4096, dropout)
        self.layer7 = vgg_fc_layer(4096, 4096, dropout)

        # Final layer
        self.layer8 = nn.Linear(4096, num_classes)

        # Initial
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


def vgg16_bn(pretrained=True, dsbn=False, num_domains=1, **kwargs):
    """Constructs a vgg_16_bn model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG16_bn(dsbn=dsbn, num_domains=num_domains, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load("/data/user21100736/huawei/V1/logs/vgg16_bn-6c64b313.pth"), False)
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
        print("########### pretrained ##############")
        mynn.forgiving_state_restore(model, torch.load("/data/user21100736/huawei/V1/logs/vgg16_bn-6c64b313.pth"))
    return model