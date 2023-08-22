import re
from typing import List, Any, Tuple

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassType, ClassFactory
from sedna.algorithms.seen_task_learning.artifact import Task

from sklearn.cluster import KMeans
import numpy as np

from dataloaders import custom_transforms as tr
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader
from torchvision import transforms
from itertools import chain

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

__all__ = ('TaskDefinitionAll',)

############## 表征计算基础模块 ##############
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

def calc_style_std_mean(target):
    input_mean, input_std = calc_mean_std(target)
    input_std=input_std.cpu()
    input_mean=input_mean.cpu()
    mean = input_mean.detach().numpy()
    std = input_std.detach().numpy()

    return np.append(mean, std)

def data_preprocess(image_urls):
    transformed_images = []
    for paths in image_urls:
        if len(paths) == 2:
            img_path, depth_path = paths
            _img = Image.open(img_path).convert('RGB')
            _depth = Image.open(depth_path)
        else:
            img_path = paths[0]
            _img = Image.open(img_path).convert('RGB')
            _depth = _img

        sample = {'image': _img, 'depth': _depth, 'label': _img}
        composed_transforms = transforms.Compose([
            # tr.CropBlackArea(),
            # tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        transformed_images.append((composed_transforms(sample), img_path))

    return transformed_images

@ClassFactory.register(ClassType.STP, alias="TaskDefinitionAll")
class TaskDefinitionAll:
    """
    Dividing datasets based on the their origins.

    Parameters
    ----------
    origins: List[Metadata]
        metadata is usually a class feature label with a finite values.
    """

    def __init__(self, **kwargs):
        self.origins = kwargs.get("origins", ["all"])
        self.splits = kwargs.get("splits")

    def __call__(self,
                 samples: BaseDataSource, **kwargs) -> Tuple[List[Task],
                                                             Any,
                                                             BaseDataSource]:
        ########## 根据app选择进行任务表征提取 or 任务划分
        if "app" in kwargs and kwargs["app"] == "embedding_extraction":
            data = data_preprocess(samples)
            data_loader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)

            if kwargs["mode"] == "eval":
                print("------------------ processing embedding extraction! ------------------")
            elif kwargs["mode"] == "train":
                print("------------------ 1. processing embedding extraction! ------------------")
            print("------------------ resnet18 is selected for extraction. ------------------")
            embedding_model = resnet18(pretrained=True, efficient=False, use_bn=True)
            embedding_model.eval()
            task_embeddings = []
            
            for sample, image_name in data_loader:
                image = sample['image']    
                _, _, embedding_feats = embedding_model(image)

                embedding_code = calc_style_std_mean(torch.unsqueeze(embedding_feats[0][0, :, :, :], 0))
                embedding_code = np.append(embedding_code, calc_style_std_mean(torch.unsqueeze(embedding_feats[1][0, :, :, :], 0)))

                task_embeddings.append([embedding_code, image_name])

            if kwargs["mode"] == "eval":
                print("------------------ embedding extraction finished! ------------------")
            elif kwargs["mode"] == "train":
                print("------------------ 1. embedding extraction finished! ------------------")

            return task_embeddings

        if "app" in kwargs and kwargs["app"] == "task_defination":
            task_embeddings = kwargs["task_embeddings"]
            task_names = []
            for i in range(self.splits):
                task_name = "task" + str(i)
                task_names.append(task_name)
            self.origins = task_names

            tasks = []
            d_type = samples.data_type
            x_data = samples.x
            y_data = samples.y
            
            task_index = dict(zip(self.origins, range(len(self.origins)))) # {'task0': 0, 'task1': 1, 'task2': 2}
            
            # 创建数据类
            style_df=[]
            for i in range(self.splits):
                style_df.append(BaseDataSource(data_type=d_type))
                style_df[i].x, style_df[i].y=[], []
                
            import pdb
            # pdb.set_trace()

            # 聚类
            embedding_prototype, embedding_index = self.embedding_mining(task_embeddings)
            
            # 分配数据
            for i in range(samples.num_examples()):
                style_df[embedding_index[i]].x.append(x_data[i])
                style_df[embedding_index[i]].y.append(y_data[i])

            # 创建任务
            tasks=[]
            for i in range(self.splits):
                g_attr='task_'+str(i)+'_model'
                task_obj = Task(entry=g_attr, samples=style_df[i], meta_attr=embedding_prototype[i])
                tasks.append(task_obj)


            return tasks, task_index, samples
        
    def embedding_mining(self, task_embeddings):
        # 提取图片的向量和路径
        vectors, _ = zip(*task_embeddings)

        # 使用 KMeans++ 进行聚类
        kmeans = KMeans(n_clusters=self.splits, init='k-means++', random_state=0).fit(vectors)

        # 获取聚类结果
        embedding_prototype = kmeans.cluster_centers_
        embedding_index = kmeans.labels_

        # 返回聚类中心向量和每张图片的所属类
        return embedding_prototype, embedding_index



############## 自定义表征提取模型 ##############
# 基础卷积模块
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def _bn_function_factory(conv, norm, relu=None):
    """return a conv-bn-relu function"""
    def bn_function(x):
        x = conv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x

    return bn_function

def do_efficient_fwd(block, x, efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block, x)
    else:
        return block(x)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

# 模型大模块
upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
batchnorm_momentum = 0.01 / 2

def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True, bn_momentum=0.1, bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module('relu', nn.ReLU(inplace=batch_norm is True))
        padding = k // 2  # same conv
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))

class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(_Upsample, self).__init__()
        print(f'Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}')
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):
        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)
        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=128,
                 grids=(6, 3, 2, 1), square_grid=False, bn_momentum=0.1, use_bn=True):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn',
                            _BNReluConv(num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i),
                                _BNReluConv(num_features, level_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))
        self.spp.add_module('spp_fuse',
                            _BNReluConv(final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
               x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, efficient=False, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

    def forward(self, x):
        residual = x

        bn_1 = _bn_function_factory(self.conv1, self.bn1, self.relu)
        bn_2 = _bn_function_factory(self.conv2, self.bn2)

        out = do_efficient_fwd(bn_1, x, self.efficient)
        out = do_efficient_fwd(bn_2, out, self.efficient)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        relu = self.relu(out)

        return relu, out

# 自定义模型
class ResNet(nn.Module):
    def __init__(self, block, layers, *, num_features=128, k_up=3, efficient=True, use_bn=True,
                 spp_grids=(8, 4, 2, 1), spp_square_grid=False, **kwargs):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.efficient = efficient
        self.use_bn = use_bn

        # rgb branch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # depth branch
        self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1_d = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu_d = nn.ReLU(inplace=True)
        self.maxpool_d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        upsamples = []
        # 修改 _make_layer_rgb  _make_layer
        self.layer1 = self._make_layer_rgb(block, 64, 64, layers[0])
        self.layer1_d = self._make_layer_d(block, 64, 64, layers[0])
        self.attention_1 = self.attention(64)
        self.attention_1_d = self.attention(64)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)] #  num_maps_in, skip_maps_in, num_maps_out, k: kernel size of blend conv

        self.layer2 = self._make_layer_rgb(block, 64, 128, layers[1], stride=2)
        self.layer2_d = self._make_layer_d(block, 64, 128, layers[1], stride=2)
        self.attention_2 = self.attention(128)
        self.attention_2_d = self.attention(128)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer3 = self._make_layer_rgb(block, 128, 256, layers[2], stride=2)
        self.layer3_d = self._make_layer_d(block, 128, 256, layers[2], stride=2)
        self.attention_3 = self.attention(256)
        self.attention_3_d = self.attention(256)
        upsamples += [_Upsample(num_features, self.inplanes, num_features, use_bn=self.use_bn, k=k_up)]

        self.layer4 = self._make_layer_rgb(block, 256, 512, layers[3], stride=2)
        self.layer4_d = self._make_layer_d(block, 256, 512, layers[3], stride=2)
        self.attention_4 = self.attention(512)
        self.attention_4_d = self.attention(512)

        self.fine_tune = [self.conv1, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4,
                          self.conv1_d, self.maxpool_d, self.layer1_d, self.layer2_d, self.layer3_d, self.layer4_d]
        if self.use_bn:
            self.fine_tune += [self.bn1, self.bn1_d, self.attention_1, self.attention_1_d, self.attention_2, self.attention_2_d,
                               self.attention_3, self.attention_3_d, self.attention_4, self.attention_4_d]

        num_levels = 3
        self.spp_size = num_features
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.spp = SpatialPyramidPooling(self.inplanes, num_levels, bt_size=bt_size, level_size=level_size,
                                         out_size=self.spp_size, grids=spp_grids, square_grid=spp_square_grid,
                                         bn_momentum=0.01 / 2, use_bn=self.use_bn)
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = []#[ self.spp, self.upsample]
        self.fine_tune += [self.spp, self.upsample]

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer_rgb(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def _make_layer_d(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            layers = [nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [block(inplanes, planes, stride, downsample, efficient=self.efficient, use_bn=self.use_bn)]
        inplanes = planes * block.expansion
        self.inplanes = inplanes
        for i in range(1, blocks):
            layers += [block(inplanes, planes, efficient=self.efficient, use_bn=self.use_bn)]

        return nn.Sequential(*layers)

    def channel_attention(self, rgb_skip, depth_skip, attention):
        assert rgb_skip.shape == depth_skip.shape, 'rgb skip shape:{} != depth skip shape:{}'.format(rgb_skip.shape, depth_skip.shape)
        # single_attenton
        rgb_attention = attention(rgb_skip)
        depth_attention = attention(depth_skip)
        rgb_after_attention = torch.mul(rgb_skip, rgb_attention)
        depth_after_attention = torch.mul(depth_skip, depth_attention)
        skip_after_attention = rgb_after_attention + depth_after_attention
        return skip_after_attention

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)


    def random_init_params(self):
        return chain(*[f.parameters() for f in self.random_init])

    def fine_tune_params(self):
        return chain(*[f.parameters() for f in self.fine_tune])

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, rgb):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        pool1 = x

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        pool2 = x
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]
        x, skip = self.forward_resblock(x.detach(), self.layer4)
        features += [self.spp.forward(skip)]
        
        embedding_feats = [pool1, pool2]
        return features, embedding_feats

    def forward_down_fusion(self, rgb, depth):
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        depth = depth.unsqueeze(1)
        y = self.conv1_d(depth)
        y = self.bn1_d(y)
        y = self.relu_d(y)
        y = self.maxpool_d(y)

        features = []
        # block 1
        x, skip_rgb = self.forward_resblock(x.detach(), self.layer1)
        y, skip_depth = self.forward_resblock(y.detach(), self.layer1_d)
        x_attention = self.attention_1(x)
        y_attention = self.attention_1_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb.detach()]
        # block 2
        x, skip_rgb = self.forward_resblock(x.detach(), self.layer2)
        y, skip_depth = self.forward_resblock(y.detach(), self.layer2_d)
        x_attention = self.attention_2(x)
        y_attention = self.attention_2_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb.detach()]
        # block 3
        x, skip_rgb = self.forward_resblock(x.detach(), self.layer3)
        y, skip_depth = self.forward_resblock(y.detach(), self.layer3_d)
        x_attention = self.attention_3(x)
        y_attention = self.attention_3_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [skip_rgb.detach()]
        # block 4
        x, skip_rgb = self.forward_resblock(x.detach(), self.layer4)
        y, skip_depth = self.forward_resblock(y.detach(), self.layer4_d)
        x_attention = self.attention_4(x)
        y_attention = self.attention_4_d(y)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        x = x + y
        features += [self.spp.forward(x)]
        return features


    def forward_up(self, features):
        features_ = features[0]
        embedding_feats = features[1]
        features = features_[::-1]

        x = features[0]

        upsamples = []
        i = 0
        for skip, up in zip(features[1:], self.upsample):
            i += 1
            #print(len(self.upsample))
            if i < len(self.upsample):
                x = up(x, skip)
            else:
                x = up(x, skip)
            upsamples += [x]
        return x, {'features': features, 'upsamples': upsamples}, embedding_feats

    def forward(self, rgb, depth = None):
        if depth is None:
            return self.forward_up(self.forward_down(rgb))
        else:
            return self.forward_up(self.forward_down_fusion(rgb, depth))

    def _load_resnet_pretrained(self, url):
        pretrain_dict = model_zoo.load_url(model_urls[url])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # print('%%%%% ', k)
            if k in state_dict:
                if k.startswith('conv1'):
                    model_dict[k] = v
                    # print('##### ', k)
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6]+'_d'+k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
        print('pretrained dict loaded sucessfully')
    return model