import torch
import torch.nn as nn

from .interface_ISNN import ISNN
from .net_RateBN import RateBatchNorm

def norm_layer_tracking_off(net):
    # 关闭norm层均值计算跟踪
    for m in net.modules():
        if isinstance(m,(torch.nn.BatchNorm2d,RateBatchNorm)):
            m.track_running_stats=False

def norm_layer_tracking_on(net):
    # 开启norm层均值计算跟踪
    for m in net.modules():
        if isinstance(m,(torch.nn.BatchNorm2d,RateBatchNorm)):
            m.track_running_stats=True

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = (torch.arange(kernel_size).reshape(-1, 1),
        torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt

    return weight

def kaiming_init(net):
    # x=list(net.modules())
    for m in net.modules():
        if(m==net): continue
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d,RateBatchNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        # elif isinstance(m,nn.ConvTranspose2d):
        #     m.weight.data=bilinear_kernel(m.in_channels,m.out_channels,m.kernel_size[0])