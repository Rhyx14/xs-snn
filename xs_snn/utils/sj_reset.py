import torch.nn as nn
from spikingjelly.activation_based import base
import logging
def reset_net(net: nn.Module):
    """
    * :ref:`API in English <reset_net-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若 ``m `` 为 ``base.MemoryModule`` 函数或者是拥有 ``reset()`` 方法，则调用 ``m.reset()``。

    * :ref:`中文API <reset_net-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset the whole network.  Walk through every ``Module`` as ``m``, and call ``m.reset()`` if this ``m`` is ``base.MemoryModule`` or ``m`` has ``reset()``.
    """
    for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                logging.warning(f'Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base'
                                f'.MemoryModule')
            m.reset()