import torch
import torch.nn as nn
class BatchNorm(nn.Module):
    def __init__(self,channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,name='',**kwargs):
        '''
        Batch Norm
        '''
        super().__init__()
        self._norm=nn.BatchNorm2d(channel,eps,momentum,affine,track_running_stats)
        self.weight=self._norm.weight.data
        self.bias=self._norm.bias.data
        self.name=name

    def forward(self,x):
        '''
        前传

        x [t,b,*size]
        '''
        t,b,c,h,w= x.shape
        rslt=[]
        for i in range(t):
            rslt.append(self._norm(x[i]))
        return torch.stack(rslt)