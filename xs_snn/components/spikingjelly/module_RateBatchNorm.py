import torch
import einops
from einops.layers.torch import Rearrange
from .module_AggregateSpikingLayer import Aggregated_Spiking_Layer
import torch.nn as nn
class RateBatchNorm(nn.Module):
    def __init__(self,channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,name='',**kwargs):
        '''
        Rate Batch Norm
        '''
        super().__init__()
        self._norm=nn.BatchNorm2d(channel,eps,momentum,affine,track_running_stats)
        self._to_t_last=Rearrange('t b c h w -> (t b) c h w')
        self.weight=self._norm.weight.data
        self.bias=self._norm.bias.data
        self.name=name

    def forward(self,x):
        '''
        前传

        x [t,b,*size]
        '''
        t,b,c,h,w= x.shape
        x=self._to_t_last(x)
        x=self._norm(x)
        x=einops.rearrange(x,'(t b) c h w -> t b c h w',t=t)
        return x

    @staticmethod
    def fuse(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)

        beta = bn.weight
        gamma = bn.bias

        if conv.bias is not None:
            b = conv.bias
        else:
            b = mean.new_zeros(mean.shape)

        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean)/var_sqrt * beta + gamma
        fused_conv = nn.Conv2d(
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
        fused_conv.weight = nn.Parameter(w)
        fused_conv.bias = nn.Parameter(b)
        return fused_conv

class Fuse_BN():
    '''
    Fuse BN to Conv2d, only support RateBatchNorm
    '''
    def __call__(self,module):
        if isinstance(module,(Aggregated_Spiking_Layer)):
            if isinstance(module._layer[-1],(nn.Conv2d)):
                _conv_layer=module._layer[-1]
                if isinstance(module._norm,(RateBatchNorm)):
                    _b2d_norm=module._norm._norm
                    with torch.no_grad():
                        _new_conv = RateBatchNorm.fuse(_conv_layer,_b2d_norm)
                    module._layer[-1]=_new_conv
                    module._norm=None