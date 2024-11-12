from collections import defaultdict
import torch
from typing import Any
class Identical_Wrapper(torch.nn.Module):
    
    ID_Map=defaultdict(lambda : 0)
    def __init__(self,hooks=None,name='idt') -> None:
        '''
        An empty module does nothing, return the original input

        hooks: list[ callable(input)]
        '''
        super().__init__()

        self.name=name
        self.id=Identical_Wrapper.ID_Map[name]
        Identical_Wrapper.ID_Map[name]+=1

        self.state_hooks=[]
        if hooks is not None:
            assert isinstance(hooks,list)
            self.state_hooks.extend(hooks)
        pass

    def __call__(self, x,*args: Any, **kwds: Any) -> Any:

        for hooks in self.state_hooks: hooks(x)
        return x
