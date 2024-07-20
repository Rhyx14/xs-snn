from collections import defaultdict
from types import NoneType
import torch
from .interface_ISNN import ISNN
from .interface_IRateNorm import IRateNorm,INonRateNorm

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

        _steps=x.shape[0]
        if self._layer is None:
            out=x
        else:
            out=[]
            for i in range(_steps):
                out.append(self._layer(x[i]))

        if self._norm is not None:
            out=self._norm(out)

        if self._neuron_model is None:
            return torch.stack(out)

        # neuron model updating
        if self._neuron_model.aggregation():
            return self._neuron_model(out)
        else:
            _rslt=[]
            for step in range(_steps):
                _rslt.append(self._neuron_model(out[step]))
            return torch.stack(_rslt)