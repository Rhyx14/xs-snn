from xs_snn.components import Aggregated_Spiking_Layer as ASL_sj
from xs_snn.components import Identical_Wrapper
from .RateBatchNorm import RateBatchNorm
from .BatchNorm import BatchNorm
from ..components.Aggregated_Spiking_Layer import Aggregated_Spiking_Layer,TimestepContainer
import torch,inspect
from xs_snn.utils.fuse_conv2d import fuse_conv2d

@torch.no_grad()
def fuse_rateBatchNorm(module):
    if not isinstance(module,ASL_sj):
        return
    _layers=module._layer
    if not isinstance(_layers,(TimestepContainer,))  \
        or (not isinstance(_layers._torch_module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d))) \
        or (not isinstance(module._norm, (RateBatchNorm,BatchNorm))):
        return
    
    _new_conv=fuse_conv2d(_layers._torch_module,module._norm._norm)
    module._layer= TimestepContainer(_new_conv)
    module._norm= Identical_Wrapper()

def check_norm(module):
    if isinstance(module,ASL_sj):
        assert isinstance(module._norm,Identical_Wrapper) or inspect.isfunction(module._norm)