import torch
import sys
sys.path.append('..')

from interface.interface_IBackwardable_SNN import IBackwardable_SNN
from components.interface_ISNN import ISNN
from components.interface_IRateNorm import IRateNorm

class Aggregate(torch.nn.Module,IBackwardable_SNN,ISNN):
    '''
    转换为聚合模式运行的SNN层
    '''

    def __init__(self,layer:torch.nn.Module,norm:torch.nn.Module=None):
        '''
        转换为聚合模式运行的SNN层
        '''
        super().__init__()
        self._layer=layer
        self._norm=norm
    
    def forward(self,x):
        '''
        前传

        x [t,b,*size]
        '''

        if self._norm is not None:
            x=self._norm(x)
            pass
        
        with torch.no_grad():
            rslt=[]
            for step in range(x.shape[0]):
                rslt.append(self._layer(x[step],step))
            return torch.stack(rslt)
    
    def backward(self,partial):
        '''
        反传

        partial [b,*size] 没有时间
        '''
        with torch.no_grad():
            x=self._layer.backward(partial)
            if self._norm is not None:
                x=self._norm.backward(x)
        return x