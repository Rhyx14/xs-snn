from collections import defaultdict
import torch
import spikingjelly.activation_based.base as SJ_Base
from spikingjelly.clock_driven.layer import SeqToANNContainer, MultiStepContainer

class Basic_Hook():
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

class Aggregated_Spiking_Layer(torch.nn.Module):
    ID_Map=defaultdict(int)
    def __init__(self,layer:SeqToANNContainer | MultiStepContainer | torch.nn.Module = None,norm:torch.nn.Module = None,neuron_model:SJ_Base.MemoryModule = None,datahook=None,name='asl'):
        '''
        hooks:  list[ callable(input)]
        '''
        super().__init__()

        self.name=name
        self.id=Aggregated_Spiking_Layer.ID_Map[self.name]
        Aggregated_Spiking_Layer.ID_Map[self.name]+=1

        self._datahooks=[]
        if datahook is not None:
            if isinstance(datahook,list):
                self._datahooks.extend(datahook)
            else:
                self._datahooks.append(datahook)
                
        self._layer=layer
        self._norm=norm
        self._neuron_model=neuron_model

    def add_hook(self,hook):
        self._datahooks.append(hook)

    def remove_hooks(self):
        self._datahooks.clear()

    def forward(self,x):
        '''
        x: [t,b,c,h,w]
        '''
        _input=x
        _layer_output=x
        _norm_output=x

        if self._layer is not None:
            _layer_output=self._layer(x)
            _out=_layer_output

        if self._norm is not None:
            _norm_output=self._norm(_layer_output)
            _out=_norm_output

        if self._neuron_model is not None:
            _out = self._neuron_model(_norm_output)

        for _datahook in self._datahooks:
            _out=_datahook(self,_input,_layer_output,_norm_output,_out)

        return _out

