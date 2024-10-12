import torch
import einops
def fuse_conv2d(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    if isinstance(conv,torch.nn.Conv2d):
        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean)/var_sqrt * beta + gamma
        fused_conv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )
    elif isinstance(conv,torch.nn.ConvTranspose2d):
        groups=conv.groups
        w = einops.rearrange(w,'(i g) out h w -> i (g out) h w',g=groups)
        w = w * (beta / var_sqrt).reshape([1, conv.out_channels, 1, 1])
        w = einops.rearrange(w,'i (g out) h w -> (i g) out h w',g=groups)
        b = (b - mean)/var_sqrt * beta + gamma
        fused_conv = torch.nn.ConvTranspose2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )

    fused_conv.weight = torch.nn.Parameter(w)
    fused_conv.bias = torch.nn.Parameter(b)
    return fused_conv

from ..components.spikingjelly.module_AggregateSpikingLayer import Aggregated_Spiking_Layer as ASL_sj
from ..components.spikingjelly.module_RateBatchNorm import RateBatchNorm as Rbn_sj
from spikingjelly.clock_driven.layer import SeqToANNContainer
@torch.no_grad()
def fuse_rateBatchNorm_sj(module):
    if not isinstance(module,ASL_sj):
        return
    _layers=module._layer
    if _layers is None or (not isinstance(_layers[0],(torch.nn.Conv2d,torch.nn.ConvTranspose2d))) or (len(_layers) !=1) or not isinstance(module._norm, Rbn_sj):
        return
    
    _new_conv=fuse_conv2d(_layers[0],module._norm._norm)
    module._layer= SeqToANNContainer(_new_conv)
    module._norm= None

def check_norm_sj(module):
    if isinstance(module,ASL_sj):
        assert module._norm is None

from ..components.xs.aggregated_spiking_layer import Aggregated_Spiking_Layer as ASL_xs
from ..components.xs.net_RateBN import RateBatchNorm as Rbn_xs
@torch.no_grad()
def fuse_rateBatchNorm_xs(module):
    if not isinstance(module,ASL_xs):
        return
    _layers=module._layer
    if _layers is None or (not isinstance(_layers,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))) or not isinstance(module._norm, Rbn_xs):
        return
    
    _new_conv=fuse_conv2d(_layers,module._norm._norm)
    module._layer= _new_conv
    module._norm= None

def check_norm_xs(module):
    if isinstance(module,ASL_xs):
        assert module._norm is None