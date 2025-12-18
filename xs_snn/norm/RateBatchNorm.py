import torch
import einops
from einops.layers.torch import Rearrange
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