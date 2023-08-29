from .interface_IRateNorm import INonRateNorm
import torch
from typing import Union

class BatchNorm(torch.nn.Module,INonRateNorm):
    def __init__(self,channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,**kwargs):
        '''
        Rate Batch Norm (only for SNN)
        '''
        super().__init__()
        self._norm=torch.nn.BatchNorm2d(channel,eps,momentum,affine,track_running_stats)
        self.weight=self._norm.weight.data
        self.bias=self._norm.bias.data

    def forward(self,x: Union[list,torch.Tensor]) -> torch.Tensor:
        '''
        前传

        x [t,b,*size]
        '''
        frame_count= len(x)
        rslt=[]
        for i in frame_count:
            rslt.append(self._norm(x[i]))  

        return torch.stack(rslt)
