"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import pandas as pd
import xlrd
import time
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import random
from torchvision.transforms import Resize
from scipy.stats import wasserstein_distance
import imageio

import sys
sys.path.append('/data/user8302433/fc/ianvs/examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/RFNet/joint_tools')

import os
paths = os.getenv('PYTHONPATH', '').split(os.pathsep)
print(paths)
from config import cfg, assert_and_infer_cfg
from datasets import sampler
from joint_tools.utils.misc import AverageMeter, prep_experiment, evaluate_eval, evaluate_eval_dur_train, fast_hist, save_best_acc
import datasets
import loss
import network
import optimizer
from network.mynn import freeze_weights, unfreeze_weights


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepWV3Plus',
                    help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                    and deepWV3Plus (backbone: WideResNet38).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=[],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=[],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--weather', type=str, default='snow',
                    help='ACDC weather choices')
parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--wt_layer', nargs='*', type=int, default=[0,0,0,0,0,0,0],
                    help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
parser.add_argument('--wt_reg_weight', type=float, default=0.0)
parser.add_argument('--relax_denom', type=float, default=2.0)
parser.add_argument('--clusters', type=int, default=50)
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--dynamic', action='store_true', default=False)

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')
parser.add_argument('--cov_stat_epoch', type=int, default=5,
                    help='cov_stat_epoch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Visualize intermediate feature')
parser.add_argument('--use_wtloss', action='store_true', default=False,
                    help='Automatic setting from wt_layer')
parser.add_argument('--use_isw', action='store_true', default=False,
                    help='Automatic setting from wt_layer')

parser.add_argument('--dsbn', type=bool, default=False,
                    help='Whether to use DSBN in the model')
parser.add_argument('--num_domains', type=int, default=1,
                    help='The number of source domains')

parser.add_argument('--mode', type=str, default=1,
                    help='The use of this code')
parser.add_argument('--split', type=str, default=1,
                    help='The split ratio of training sets and test sets')
parser.add_argument('--spliting', type=str, default=1,
                    help='The spliting data of tasks')
parser.add_argument('--spliting_method', type=str, default=1,
                    help='The spliting method of tasks')

parser.add_argument('--storage_path', type=str, default=1,
                    help='The task-specific weights and mean embeddings of splits')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

freeze_bn_num = 0

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)

for i in range(len(args.wt_layer)):
    if args.wt_layer[i] == 1:
        args.use_wtloss = True
    if args.wt_layer[i] == 2:
        args.use_wtloss = True
        args.use_isw = True

def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    ncols = table.ncols  # 列数
    datamatrix = np.zeros((nrows, ncols))
    for i in range(nrows):
        rows = table.row_values(i)
        datamatrix[i,:] = rows
    return datamatrix

def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    # train_loader, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)
    train_loaders, val_loaders, train_obj, extra_val_loaders, covstat_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    # net = network.get_net(args, criterion, criterion_aux)

    # optim, scheduler = optimizer.get_optimizer(args, net)

    # 这里主要是在处理多GPU分布式执行的东西（dataparallel with syncBN），这有一个不好的一点 —— 多卡训练的syncBN不能装载到单卡推理的BN上，所以要用如下模块才能实现推理
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net) # convert all attr:`BatchNorm*D` layers in the model to class:`torch.nn.SyncBatchNorm` layers
    # net = network.warp_network_in_dataparallel(net, args.local_rank) # DistributedDataParallel

    epoch = 0
    i = 0
    storage_num = 0
    ################################################################### 测试
    if args.mode == "test":
        ########## 装载权重模块
        storage_path = args.storage_path
        
        snapshots = {}
        nets = {}
        optims = {}
        schedulers = {}
        avg_style_codes = {}
        for file in os.listdir(storage_path):
            if '.pth' in file:
                file = os.path.join(storage_path, file)
                style_file = os.path.join(storage_path, (file.split('/')[-1].split('.')[0] + '.xlsx'))
                storage_name = "net" + str(storage_num)
                
                avg_style_codes[storage_name] = excel2matrix(style_file)
                
                snapshots[storage_name] = file
                nets[storage_name] = network.get_net(args, criterion, criterion_aux)
                optims[storage_name], schedulers[storage_name] = optimizer.get_optimizer(args, nets[storage_name])
                epoch, mean_iu = optimizer.load_weights(nets[storage_name], optims[storage_name], schedulers[storage_name],
                                    snapshots[storage_name], args.restore_optimizer)
                
                storage_num = storage_num + 1
            epoch = 0

        ########## 测试模块
        print("#### iteration", i)
        torch.cuda.empty_cache()
        
        if len(val_loaders) == 1 or (len(val_loaders) != 1 and "research_campuse1" == args.dataset[0]):
            # Run validation only one time - To save models
            for dataset, val_loader in val_loaders.items():
                validate(val_loader, dataset, nets, criterion_val, optims, schedulers, epoch, writer, i, avg_style_codes, save_pth=False)
        else:
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, nets, optims, schedulers, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

        for dataset, val_loader in extra_val_loaders.items():
            print("Extra validating... This won't save pth file")
            validate(val_loader, dataset, nets, criterion_val, optims, schedulers, epoch, writer, i, avg_style_codes, save_pth=False)



def validate(val_loader, dataset, nets, criterion, optims, schedulers, curr_epoch, writer, curr_iter, avg_style_codes, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []
    tmp_for_save = ['image_idx', 'image_name', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'light', 
                    'sign', 'vegetation', 'terrain', 'sky', 'pedestrain', 'rider', 'car', 'truck', 'bus', 
                    'train', 'motocycle', 'bicycle', 'stair', 'curb', 'ramp', 'runway', 'flowerbed', 'door', 
                    'CCTV camera', 'Manhole', 'hydrant', 'belt', 'dustbin', 'ignore', 'miou']

    # resize_func_input = Resize([1, 3, args.crop_size, 1242])
    # resize_func_gt = Resize([1, args.crop_size, 1242])

    for i in range(len(nets)):
        storage_name = "net" + str(i)
        nets[storage_name].eval() # 切换至验证模式，仅对Dropout或BatchNorm等模块有任何影响
    # nets['net_origin'].eval() 

    for val_idx, data in enumerate(val_loader): # 一张一张来的
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image           = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data
        
        gt_image_array = gt_image.numpy()

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        # Resize

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda() # 必须要手动，使数据在GPU上进行运算

        domain_label = 0 * torch.ones(inputs.shape[0], dtype=torch.long)

        style_codes = {}
        outputs = {}
        distances = {}
        min_distance = 1
        min_distance_num = 0
        # net = nets['net_origin']
        # outputs['net_origin'], style_codes['net_origin'] = net(inputs, dsbn=args.dsbn, mode='val', domain_label=domain_label) # 推理
        _, current_style_code = nets['net0'](inputs, dsbn=args.dsbn, mode='val', domain_label=domain_label) # 获取style_codes
        current_style_code = current_style_code[:256]
        for i in range(len(nets)):
            storage_name = "net" + str(i)
            # 计算style_code的wasserstein_distance
            avg_style_code = avg_style_codes[storage_name][0][:256]
            distances[storage_name] = wasserstein_distance(current_style_code, avg_style_code)
            if distances[storage_name] < min_distance:
                min_distance = distances[storage_name]
                min_distance_num = i
        
        net = nets["net" + str(min_distance_num)]
        # if min_distance_num == 2:
        #     print(img_names)
        output, _ = net(inputs, dsbn=args.dsbn, mode='val', domain_label=domain_label)
        
        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        if "campuse1" == args.dataset[0]:
            assert output.size()[1] == 10
        elif "new_campuse1" == args.dataset[0]:
            assert output.size()[1] == 31
        else:
            assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 30 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        if "campuse1" == args.dataset[0]:
            num_classes = 10
        elif "new_campuse1" == args.dataset[0]:
            num_classes = 31
        else:
            num_classes = datasets.num_classes

        iou_current = fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(), num_classes)
        if val_idx == 0:
            iou_acc = iou_current
        else:
            iou_acc = iou_acc + iou_current
        # iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
        #                      num_classes)
        
        # 记录逐样本精度
        iu, mean_iu = record_sample(val_idx, img_names, iou_current)
        tmp = [val_idx, img_names]
        for idx in range(num_classes):
            tmp.append(iu[idx])
        tmp.append(mean_iu)
        tmp_for_save = np.vstack((tmp_for_save, tmp))
        
        del output, val_idx, data, iou_current, iu, mean_iu, tmp

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()
    
    if args.local_rank == 0:
        iu, mean_iu = evaluate_eval(args, nets['net0'], optims['net0'], schedulers['net0'], val_loss, iou_acc, dump_images,
                                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)

    # 记录测试集总精度
    tmp = ['total', 'total']
    for idx in range(num_classes):
        tmp.append(iu[idx])
    tmp.append(mean_iu)
    tmp_for_save = np.vstack((tmp_for_save, tmp))

    # 保存记录精度
    tmp_for_save = pd.DataFrame(tmp_for_save)
    _writer = pd.ExcelWriter('./rfnet_31_Research_CampusE1_' + args.spliting_method + "_" + 'joint_infer.xlsx') # 写入Excel文件
    tmp_for_save.to_excel(_writer,  float_format='%.4f', index=False, header=False)		
    _writer.save()
    _writer.close()

    return val_loss.avg

def record_sample(val_idx, img_names, hist):
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    #logging.info("image_id: {}, image_name: {}".format(val_idx, img_names))
    for idx, i in enumerate(iu):
        # Format all of the strings:
        idx_string = "{:2d}".format(idx)
        iu_string = '{:5.1f}'.format(i * 100)
    #    logging.info("label_id: {}, iU: {}".format(idx_string, iu_string))
        
    mean_iu = np.nanmean(iu)
    #logging.info('mean iU: {}'.format(mean_iu))
    
    return iu, mean_iu

def validate_for_cov_stat(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    # net.train()#eval()
    net.eval()

    for val_idx, data in enumerate(val_loader):
        img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
        img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

        with torch.no_grad():
            net([img_photometric, img_or], cal_covstat=True)

        del img_or, img_photometric, img_geometric

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / 100", val_idx + 1)
        del data

        if val_idx >= 499:
            return


def visualize_matrix(writer, matrix_arr, iteration, title_str):
    stage = 'valid'

    for i in range(len(matrix_arr)):
        C = matrix_arr[i].shape[1]
        matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
        matrix = torch.clamp(torch.abs(matrix), max=1)
        matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                        torch.abs(matrix - 1.0)), 0)
        matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
        writer.add_image(stage + title_str + str(i), matrix, iteration)


def save_feature_numpy(feature_maps, iteration):
    file_fullpath = '/home/userA/projects/visualization/feature_map/'
    file_name = str(args.date) + '_' + str(args.exp)
    B, C, H, W = feature_maps.shape
    for i in range(B):
        feature_map = feature_maps[i]
        feature_map = feature_map.data.cpu().numpy()   # H X D
        file_name_post = '_' + str(iteration * B + i)
        np.save(file_fullpath + file_name + file_name_post, feature_map)


if __name__ == '__main__':
    main()

# ssh -L 6123:127.0.0.1:6123；tensorboard --logdir="./logs/log_path/" --port=6123