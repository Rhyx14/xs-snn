import torch
import einops

class Repeat(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._args=args
        self._kwargs=kwargs
    
    def forward(self,x):
        return einops.repeat(x,*self._args,**self._kwargs)

    def extra_repr(self):
        return f'{self._args},{self._kwargs}'