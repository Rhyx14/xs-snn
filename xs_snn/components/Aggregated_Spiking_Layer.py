from collections import defaultdict
from typing import Any
import torch
import spikingjelly.activation_based.base as SJ_Base
from spikingjelly.clock_driven.layer import SeqToANNContainer, MultiStepContainer
from .Data_Hook_Component import DataHookComponent

class Aggregated_Spiking_Layer(torch.nn.Module,DataHookComponent):
    class BasicHook():
        '''
        Basic hook, recording the input/output
        '''
        def __call__(self, asl_layer_obj,input,layer_output,norm_output,output):
            self.asl_layer_obj=asl_layer_obj
            self.input=input
            self.layer_output=layer_output
            self.norm_output=norm_output
            self.output=output
            return output

    ID_Map=defaultdict(int)
    def __init__(self,
                 layer:SeqToANNContainer | MultiStepContainer | torch.nn.Module,
                 norm:torch.nn.Module,
                 neuron_model:SJ_Base.MemoryModule | torch.nn.Module,
                 datahook: list | Any=None,
                 name : str = 'asl'):
        '''
        hooks:  list[ callable(input)]
        '''
        super().__init__()
        DataHookComponent.__init__(self,datahook)

        self.name=name
        self.id=Aggregated_Spiking_Layer.ID_Map[self.name]
        Aggregated_Spiking_Layer.ID_Map[self.name]+=1

        self._layer=lambda x : x
        self._norm =lambda x : x
        self._neuron_model = lambda x: x
        
        if layer is not None: self._layer=layer
        if norm is not None: self._norm=norm
        if neuron_model is not None: self._neuron_model=neuron_model

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        _layer_output=self._layer(x)
        _norm_output=self._norm(_layer_output)
        _out = self._neuron_model(_norm_output)

        for _datahook in self._datahooks:
            _out=_datahook(self,x,_layer_output,_norm_output,_out)

        return _out

