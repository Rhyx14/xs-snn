# from https://zhuanlan.zhihu.com/p/267535838?ivk_sa=1024320u
import torch
from torchvision import models
import torch.nn.modules as modules
from spikingjelly.activation_based.base import MemoryModule
def split_parameters_for_SNN(module):
    '''
    get params of SNN

    @return : decay, // including CONV, Linear,
            : no_decay, // including BN, bias of each layer
    '''

    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif isinstance(m, MemoryModule):
            params_no_decay.extend([*m.parameters()])

        elif hasattr(m,'split_parameters'):
            _decay,_no_decay=m.split_parameters()
            params_decay.extend([*_decay])
            params_no_decay.extend([*_no_decay])

    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay