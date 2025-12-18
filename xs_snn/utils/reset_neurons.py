import torch.nn as nn
import logging
from xs_snn.components import NeuronBase
def reset_neurons(net: nn.Module):
    """
    Reset the neurons in the entire network.  Walk through every ``Module`` as ``m``, and call ``m.reset()`` if this ``m`` is ``NueronBase``.
    """
    for m in net.modules():
        if hasattr(m, 'reset') and isinstance(m,NeuronBase):
            m.reset()