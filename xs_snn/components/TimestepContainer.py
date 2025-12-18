import torch
class TimestepContainer(torch.nn.Module):
    def __init__(self,torch_module:torch.nn.Module,stateless=True):
        super().__init__()
        self._torch_module=torch_module
        if stateless:
            self.forward=self._stateless_forward
        else:
            self.forward=self._multi_step_forward
        pass

    def _stateless_forward(self,x:torch.Tensor):
        t=x.shape[0]
        x=x.flatten(0,1)
        x=self._torch_module(x)
        return x.view(t,-1,x.shape[1:])
    
    def _multi_step_forward(self,x:torch.Tensor):
        t=x.shape[0]
        rslt=[]
        for _ in range(t):
            x=self._torch_module(x)
            rslt.append(x)
        return torch.stack(rslt,dim=0)