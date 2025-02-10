from collections import defaultdict
from types import NoneType
from ..Identical_Wrapper import Identical_Wrapper
from typing import Any
import torch
import spikingjelly.activation_based.base as SJ_Base
from spikingjelly.clock_driven.layer import SeqToANNContainer, MultiStepContainer
class Aggregated_Spiking_Layer(torch.nn.Module):
    ID_Map=defaultdict(int)
    def __init__(self,layer:SeqToANNContainer | MultiStepContainer = None,norm:torch.nn.Module = None,neuron_model:SJ_Base.MemoryModule = None,input_hooks=None,name='asl'):
        '''
        hooks:  list[ callable(input)]
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

        # assert isinstance(self._layer,(SeqToANNContainer,MultiStepContainer,NoneType))
        
        assert isinstance(self._neuron_model,(SJ_Base.MemoryModule,Identical_Wrapper,NoneType))

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        for hooks in self.input_hooks: x=hooks(x)

        out = x
        if self._layer is not None:
            out=self._layer(out)

        if self._norm is not None:
            out=self._norm(out)

        if self._neuron_model is not None:
            out = self._neuron_model(out)
        self.output_shape=out.shape

        return out

