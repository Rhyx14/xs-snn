import torch
from ..components.NeuronBase import NeuronBase

class ReLU6(NeuronBase):
    '''
    ReLU6
    '''
    id=0
    def __init__(self,small_id=0,**kwargs):
        super().__init__()

        # serial number
        self.id=ReLU6.id+small_id
        ReLU6.id+=1

    def forward(self,input):
        rslt=torch.nn.functional.relu6(input,inplace=False)
        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks: hooks(self.id,input,rslt)
        return rslt

    def extra_repr(self):
        s = f'{super().extra_repr()},id={self.id}'
        return s
