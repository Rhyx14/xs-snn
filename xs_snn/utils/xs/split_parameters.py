# from https://zhuanlan.zhihu.com/p/267535838?ivk_sa=1024320u
import torch
from ...components.xs.Interface_ISNN import ISNN
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
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm,ISNN)):
            params_no_decay.extend([*m.parameters()])

        elif hasattr(m,'split_parameters'):
            _decay,_no_decay=m.split_parameters()
            params_decay.extend([*_decay])
            params_no_decay.extend([*_no_decay])
        else:
            for _d in m.__dict__:
                if isinstance(_d,torch.nn.Parameter):
                    params_no_decay.append(_d)

    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay