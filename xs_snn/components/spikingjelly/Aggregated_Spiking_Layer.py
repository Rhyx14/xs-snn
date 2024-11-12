from collections import defaultdict
from types import NoneType
from typing import Any
import torch
import spikingjelly.activation_based.base as SJ_Base
from spikingjelly.clock_driven.layer import SeqToANNContainer, MultiStepContainer
class Aggregated_Spiking_Layer(torch.nn.Module):
    ID_Map=defaultdict(int)
    def __init__(self,layer:SeqToANNContainer | MultiStepContainer = None,norm:torch.nn.Module = None,neuron_model:SJ_Base.MemoryModule = None,hooks=None,name='asl'):
        '''
        hooks:  list[ callable(input)]
        '''
        super().__init__()

        self.name=name
        self.id=Aggregated_Spiking_Layer.ID_Map[self.name]
        Aggregated_Spiking_Layer.ID_Map[self.name]+=1

        self.state_hooks=[]
        if hooks is not None:
            assert isinstance(hooks,list)
            self.state_hooks.extend(hooks)
                
        self._layer=layer
        self._norm=norm
        self._neuron_model=neuron_model

        assert isinstance(self._layer,(SeqToANNContainer,MultiStepContainer,NoneType))
        
        assert isinstance(self._neuron_model,(SJ_Base.MemoryModule,NoneType))

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        for hooks in self.state_hooks: hooks(x)

        out = x
        if self._layer is not None:
            out=self._layer(out)

        if self._norm is not None:
            out=self._norm(out)

        if self._neuron_model is not None:
            out = self._neuron_model(out)
        self.output_shape=out.shape

        return out

