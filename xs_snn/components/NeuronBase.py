import torch
class NeuronBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state_hooks=[]

    def forward(self,x):
        t=x.shape[0]
        rslt=[]
        for _t in range(t):
            _tmp=self.single_step_forward(x[_t])
            rslt.append(_tmp)
        return torch.stack(rslt,dim=0)
    
    def single_step_forward(self,input):
        raise NotImplementedError
    
    def reset(self):
        pass