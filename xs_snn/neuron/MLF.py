import torch
import torch.nn.functional as F
from ..components.NeuronBase import NeuronBase
from xs_snn.utils.override import Override

class LIF2_dynamic(torch.autograd.Function):
    """ 
    LIF update, with arctan surrogate gradient
    """
    @staticmethod
    def forward(ctx, s_psp,vth,tau):
        ctx.save_for_backward(s_psp)
        ctx.vth=vth
        ctx.tau=tau
        output,U= _lif_forward(s_psp,vth,tau)
        return output.type_as(s_psp), U

    @staticmethod
    def backward(ctx, dy,du):
        s_psp, = ctx.saved_tensors 
        return _lif_back(s_psp, dy,du,ctx.vth,ctx.tau),None,None
        # frac=(3.1416 /2 * alpha)**2
        # hu = alpha / (2 * (1 + input**2 * frac))
        # return dy * hu

# @torch.jit.script
def _lif_forward(s_psp:torch.Tensor,vth:float,tau:float):
    output = torch.gt(s_psp, vth).type_as(s_psp)
    U=  tau * (1-output) * s_psp
    return output,U

@torch.jit.script
def _lif_back(s_psp,dy,du,vth:float,tau:float):
    hu = 1 / (1+((s_psp-vth)*3.1415926)**2)
    do=dy*hu # gradient from surrogate, space
    dt= tau * torch.where(s_psp<=vth,du,0) # gradient from membrane potential, time
    return do+dt

class MLF(NeuronBase):
    '''
    MLF

    vth step is according to 
    https://arxiv.org/pdf/2210.06386.pdf

    tau 
    https://github.com/langfengQ/MLF-DSResNet/blob/main/parallel_nets/spike_layer_for_cifar10.py
    '''
    duplicate=3
    id=0
    leak_mode='mlf'
    reset_mode=None

    def __init__(self,delay=None,duplicate=None,leak_mode=None,vth_step=1.,vth_base=0.6,small_id=0,**kwargs):
        super().__init__()
        # serial number
        self.id=MLF.id+small_id
        MLF.id+=1

        Override(MLF,self,'duplicate',duplicate)
        Override(MLF,self,'leak_mode',leak_mode)
        
        self.vth_step=vth_step
        self.vth_base=vth_base  
        self.delay=delay

        if self.leak_mode=='smooth_exclusive':
            self.tau=[0.1 for i in range(self.duplicate)]
            self.tau[0]=1
        elif self.leak_mode=='exclusive':
            self.tau=[0 for i in range(self.duplicate)]
            self.tau[0]=1
        elif self.leak_mode=='decay':
            self.tau=[0.01/(i+1) for i in range(self.duplicate)]
            self.tau[0]=1
        elif self.leak_mode=='mlf':
            self.tau=[0.25 for i in range(self.duplicate)]
        elif self.leak_mode=='zero':
            self.tau=[0 for i in range(self.duplicate)]
        else:
            raise NotImplementedError

        self.Vth=[vth_base+vth_step*i for i in range(self.duplicate)]
        self.reset()

    def reset(self):
        self.U=[None for i in range(self.duplicate)]
        
    def single_step_forward(self,input):
        rslt=[]
        for i in range(self.duplicate):
            rslt.append(self._update(input,i))

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,input,sum(rslt))
        return sum(rslt)

    def _update(self,input,d_id):
        if(self.U[d_id] is None):
            self.U[d_id]=input
        else:
            # t=torch.zeros(*input.shape)+self.U
            self.U[d_id]=self.U[d_id] + input

        o,self.U[d_id] = LIF2_dynamic.apply(self.U[d_id],self.Vth[d_id],self.tau[d_id])
        return o

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},duplicate={self.duplicate},leak_mode={self.leak_mode},vth_base={self.vth_base},vth_step={self.vth_step},reset_mode={MLF.reset_mode}')
        return s

class LIF(MLF):
    def __init__(self, delay=None, step_mode='m', vth_base=0.6, small_id=0, **kwargs):
        super().__init__(delay, 1, 'mlf', 1, step_mode, vth_base, small_id, **kwargs)

    def extra_repr(self):
        s = (f'delay={self.delay},id={self.id},vth={self.vth_base}')
        return s