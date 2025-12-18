import copy,torch,math
import torch.nn as nn
import numpy as np
from ..components.NeuronBase import NeuronBase
ASGL_CONFIG={
    'wd': 5e-4,
    'act': 'spike',
    'alpha': 1.0,
    'use_gate': False, 
    'granularity':'layer',
    'decay': 0.5,
    'p': 0.1,
    'gamma': 1, 
    'train_decay': True,
    'train_thresh': False,
    'means': 1.0,
    'lamb': 1e-3,
    'train_width': True  
}
class ASGL(NeuronBase):
    """
        simulating iterative leaky-integrate-and-fire neurons and mapping input currents into output spikes
    """
    @staticmethod
    def warp_decay(decay):
        return torch.tensor(math.log(decay / (1 - decay)))

    def __init__(self, thresh=0.5,vreset=None, use_gate=False,**kwds):
        super(ASGL, self).__init__()
        self.vreset = copy.deepcopy(vreset)
        self.use_gate = use_gate
        self.decay = nn.Parameter(ASGL.warp_decay(ASGL_CONFIG['decay'])) if ASGL_CONFIG['train_decay'] else ASGL.warp_decay(ASGL_CONFIG['decay'])
        self.thresh = nn.Parameter(torch.tensor(thresh)) if ASGL_CONFIG['train_thresh'] else thresh
        self.spike_fn = EfficientNoisySpikeII(p=ASGL_CONFIG['p'], 
                                              inv_sg=InvRectangle(alpha=ASGL_CONFIG['alpha'],learnable=ASGL_CONFIG['train_width'],granularity=ASGL_CONFIG['granularity']),
                                            spike=True)
        self.vmem=0.

    def single_step_forward(self, spsp):
        # print(self.thresh)
        # print(F.sigmoid(self.decay))
        # if isinstance(self.spike_fn, EfficientNoisySpike):
        #     psp /= self.spike_fn.inv_sg.alpha
        # print('teste')
        gates = None
        # if self.use_gate:
        #     psp, gates = torch.chunk(psp, 2, dim=1)
            # gates = torch.sigmoid(gates)
        self.vmem = torch.sigmoid(self.decay) * self.vmem + spsp
        if isinstance(self.spike_fn, EfficientNoisySpikeII):  # todo: check here
            # print('trigger!')
            self.spike_fn.reset_mask()
        if self.use_gate:
            spike = self.spike_fn(self.vmem - self.thresh, gates)
        else:
            spike = self.spike_fn(self.vmem - self.thresh)
        if self.vreset is None:
            self.vmem -= self.thresh * spike
        else:
            self.vmem = self.vmem * (1 - spike) + self.vreset * spike
        # spike *= self.thresh
        return spike

    def reset(self):
        self.vmem = 0.
        # if isinstance(self.decay, nn.Parameter):
        #     self.decay.data.clamp_(0., 1.)
        if isinstance(self.thresh, nn.Parameter):
            self.thresh.data.clamp_(min=0.)

class InvSigmoid(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True):
        super(InvSigmoid, self).__init__()
        self.learnable = learnable
        self.alpha = alpha

    def get_temperature(self):
        return self.alpha.detach().clone()

    def forward(self, x, gates=None):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
        if gates is None:
            return torch.sigmoid(self.alpha * x)
        else:
            raise NotImplementedError('gates is not supported now')


class InvRectangle(nn.Module):
    def __init__(self, alpha: float = 1.0, learnable=True, granularity='layer'):
        super(InvRectangle, self).__init__()
        self.granularity = granularity
        self.learnable = learnable
        self.alpha = np.log(alpha) if learnable else torch.tensor(np.log(alpha))

    def get_temperature(self):
        if self.granularity != "layer":
            return self.alpha.detach().mean().reshape([1])
        else:
            if isinstance(self.alpha, nn.Parameter):
                return self.alpha.detach().clone()
            else:
                return torch.tensor([self.alpha])

    def forward(self, x, gates=None):
        if self.learnable and not isinstance(self.alpha, nn.Parameter):
            if self.granularity == 'layer':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device))
            elif self.granularity == 'channel':
                self.alpha = nn.Parameter(torch.Tensor([self.alpha]).to(x.device)) if x.dim() <= 2 else nn.Parameter(
                    torch.ones(1, x.shape[1], 1, 1, device=x.device) * self.alpha)
            elif self.granularity == 'neuron':
                self.alpha = nn.Parameter(torch.ones_like(x[0]) * self.alpha)
            else:
                raise NotImplementedError('other granularity is not supported now')
            # print(self.alpha.shape)
            # self.alpha = nn.Parameter(torch.ones_like(x[0]) * self.alpha) if self.neuron_wise else nn.Parameter(
            #     torch.Tensor([self.alpha]).to(x.device))
        if gates is None:
            return torch.clamp(torch.exp(self.alpha) * x + 0.5, 0, 1.0)
        else:
            return torch.clamp(torch.exp(gates) * x + 0.5, 0, 1.0)


class EfficientNoisySpike(nn.Module):
    def __init__(self, inv_sg=InvSigmoid()):
        super(EfficientNoisySpike, self).__init__()
        self.inv_sg = inv_sg

    def forward(self, x, gates=None):
        return self.inv_sg(x, gates) + ((x >= 0).float() - self.inv_sg(x, gates)).detach()


class EfficientNoisySpikeII(EfficientNoisySpike):  # todo: write ABC
    def __init__(self, inv_sg, p, spike):
        super(EfficientNoisySpikeII, self).__init__()
        self.inv_sg = inv_sg
        self.p = p
        self.spike = spike
        self.reset_mask()

    def create_mask(self, x: torch.Tensor):
        return torch.bernoulli(torch.ones_like(x) * (1 - self.p))

    def forward(self, x, gates=None):
        sigx = self.inv_sg(x, gates)
        if self.training:
            if self.mask is None:
                self.mask = self.create_mask(x)
            return sigx + (((x >= 0).float() - sigx) * self.mask).detach()
            # return sigx * (1 - self.mask) + ((x >= 0).float() * self.mask).detach()
        if self.spike:
            return (x >= 0).float()
        else:
            return sigx

    def reset_mask(self):
        self.mask = None
