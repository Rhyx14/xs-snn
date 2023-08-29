from .interface_ISNN import ISNN
from .interface_IRateNorm import IRateNorm
import torch
import einops
from einops.layers.torch import Rearrange

class RateBatchNorm(torch.nn.Module,IRateNorm):
    def __init__(self,channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,**kwargs):
        '''
        Rate Batch Norm
        '''
        super().__init__()
        self._norm=torch.nn.BatchNorm2d(channel,eps,momentum,affine,track_running_stats)
        self._to_t_last=Rearrange('t b c h w -> (t b) c h w')
        self.weight=self._norm.weight.data
        self.bias=self._norm.bias.data

    def forward(self,x):
        '''
        前传

        x [t,b,*size]
        '''
        if isinstance(x,list):
            x=torch.stack(x)
        t,b,c,h,w= x.shape
        x=self._to_t_last(x)
        x=self._norm(x)
        x=einops.rearrange(x,'(t b) c h w -> t b c h w',t=t)    

        return x
