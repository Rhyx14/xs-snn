from collections import defaultdict
from types import NoneType
import torch
from .interface_ISNN import ISNN
from .interface_IRateNorm import IRateNorm,INonRateNorm

from spikingjelly.clock_driven.functional import multi_step_forward

class Aggregated_Spiking_Layer(torch.nn.Module):
    '''
    转换为聚合模式运行的SNN层,auto grad, with neuron model
    '''
    id=0
    def __init__(self,layer:torch.nn.Module,norm:torch.nn.Module=None,neuron_model:ISNN=None,hooks=None,name='asl'):
        '''
        转换为聚合模式运行的SNN层

        hooks: 获取输入 list[ callable(input)]

        // layer 必须是ANN层
        '''
        super().__init__()

        self.comments=name
        self.id=Aggregated_Spiking_Layer.id
        Aggregated_Spiking_Layer.id+=1

        self.state_hooks=[]
        if hooks is not None:
            assert isinstance(hooks,list)
            self.state_hooks.extend(hooks)
                
        self._layer=layer
        self._norm=norm
        self._neuron_model=neuron_model
        assert isinstance(self._neuron_model,(ISNN,NoneType))
        assert isinstance(self._norm,(IRateNorm,INonRateNorm,NoneType))

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        for hooks in self.state_hooks: hooks(x)

        out=multi_step_forward(x,self._layer)

        if self._norm is not None:
            out=self._norm(out)

        if self._neuron_model is None:
            return out

        # neuron model updating
        if self._neuron_model.aggregation():
            return self._neuron_model(out)
        else:
            return multi_step_forward(out,self._neuron_model)
            
class Aggregated(torch.nn.Module):
    '''
    时间聚合模式运行,auto grad
    '''
    id_map=defaultdict(lambda : 0)
    def __init__(self,layer:torch.nn.Module,hooks=None,name='agg'):
        '''
        时间聚合模式运行

        hooks: 获取输入 list[ callable(input)]

        '''
        super().__init__()
        self.comments=name
        self.id=Aggregated.id_map[name]
        Aggregated.id_map[name]+=1

        self._layer=layer

        self.state_hooks=[]
        if hooks is not None:
            assert isinstance(hooks,list)
            self.state_hooks.extend(hooks)
    
    def forward(self,x):
        '''
        前传

        x [t,b,*size]
        '''

        for hooks in self.state_hooks: hooks(x)

        if isinstance(self._layer,(ISNN,)) and self._layer.aggregation():
            return self._layer(x)
        else:
            return multi_step_forward(x,self._layer)