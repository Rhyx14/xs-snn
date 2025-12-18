import torch
from ..components.NeuronBase import NeuronBase

class ReLU(NeuronBase):
    '''
    ReLU
    '''
    id=0
    def __init__(self,small_id=0,**kwargs):
        super().__init__()

        # serial number
        self.id=ReLU.id+small_id
        ReLU.id+=1

    def forward(self,input):

        rslt=torch.nn.functional.relu(input,inplace=False)
        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks: hooks(self.id,input,rslt)
        return rslt

    def extra_repr(self):
        s = f'{super().extra_repr()},id={self.id}'
        return s
