from types import NoneType
from collections import defaultdict
import torch
from .Interface_ISNN import ISNN
class Aggregated_Spiking_Layer(torch.nn.Module):
    '''
    转换为聚合模式运行的SNN层,auto grad, with neuron model
    '''
    ID_Map=defaultdict(int)
    def __init__(self,layer:torch.nn.Module,norm:torch.nn.Module=None,neuron_model:ISNN=None,input_hooks=None,name='asl'):
        '''
        转换为聚合模式运行的SNN层

        hooks: 获取输入 list[ callable(input)]

        // layer 必须是ANN层
        '''
        super().__init__()

        self.name=name
        self.id=Aggregated_Spiking_Layer.ID_Map[self.name]
        Aggregated_Spiking_Layer.ID_Map[self.name]+=1

        self.input_hooks=[]
        if input_hooks is not None:
            assert isinstance(input_hooks,list)
            self.input_hooks.extend(input_hooks)
                
        self._layer=layer
        self._norm=norm
        self._neuron_model=neuron_model
        assert isinstance(self._neuron_model,(ISNN,NoneType))

        self._delay=False
        
    def _to_delay_mode(self):
        self._delay=True

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        for hooks in self.input_hooks: x=hooks(x)

        _steps=x.shape[0]
        if self._layer is None:
            out=x
        else:
            out=[]
            if self._delay:
                for i in range(1,_steps):
                    out.append(self._layer(x[i-1]))
                out.insert(0,torch.zeros_like(out[0]))
            else:
                for i in range(_steps):
                    out.append(self._layer(x[i]))
            out=torch.stack(out).contiguous()

        if self._norm is not None:
            out=self._norm(out)

        if self._neuron_model is None:
            return out

        # neuron model updating
        if self._neuron_model.aggregation():
            return self._neuron_model(out)
        else:
            _rslt=[]
            for step in range(_steps):
                self._neuron_model.update_states({'current_step':step,'delay':self._delay})
                _rslt.append(self._neuron_model(out[step]))
            return torch.stack(_rslt)
 
    class Delay():
        '''
        set spiking neuron with 1 delay
        '''
        def __call__(self,module):
            if isinstance(module,Aggregated_Spiking_Layer):
                module._to_delay_mode()
            pass