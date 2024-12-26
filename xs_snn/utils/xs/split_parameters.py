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
    for _m in module.modules():
        if isinstance(_m, torch.nn.Linear):
            params_decay.append(_m.weight)
            if _m.bias is not None:
                params_no_decay.append(_m.bias)
        elif isinstance(_m, torch.nn.modules.conv._ConvNd):
            params_decay.append(_m.weight)
            if _m.bias is not None:
                params_no_decay.append(_m.bias)
        elif isinstance(_m, (torch.nn.modules.batchnorm._BatchNorm,ISNN)):
            params_no_decay.extend([*_m.parameters()])

        elif hasattr(_m,'split_parameters'):
            _decay,_no_decay=_m.split_parameters()
            params_decay.extend([*_decay])
            params_no_decay.extend([*_no_decay])
        else:
            for _2_param in _m._parameters.values():
                params_no_decay.append(_2_param)

    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay