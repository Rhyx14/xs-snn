'''
Adapted from
https://github.com/langfengQ/MLF-DSResNet/blob/main/parallel_nets/spike_layer_for_cifar10.py
to spikingjelly version
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..components.NeuronBase import NeuronBase

# Vth = 0.6
# Vth2 = 1.6
# Vth3 = 2.6

TAU = 0.25
from .surrogate_functions import G_arctan
spikefunc=G_arctan.apply

class Original_MLF(NeuronBase):
    """ MLF unit.
    """
    id=0
    def __init__(self,duplicate=3,small_id=0,**kwds):
        super().__init__()
        self.duplicate=duplicate
        self.vths=[0.6 + i for i in range(self.duplicate)]

        self.id=Original_MLF.id+small_id
        Original_MLF.id+=1

    def forward(self, x):
        frame_count,bs=x.shape[0],x.shape[1] # assert x.shape [t b c h w] or [t b c]
        u= [torch.zeros(x.shape[1:], device=x.device) for i in range(self.duplicate)]
        o = torch.zeros(x.shape, device=x.device)

        for _t in range(frame_count):
            for i in range(self.duplicate):
                u[i] = TAU * u[i] * (1- spikefunc(u[i],self.vths[i]).detach()) + x[_t]
                _o=spikefunc(u[i],self.vths[i])
                o[_t,...] += _o

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,x,o)
        return o

    def extra_repr(self):
        return super().extra_repr() + f'duplicate (K)= {self.duplicate}, vth_base={self.vths[0]}'