import torch
class G_arctan(torch.autograd.Function):
    """ 
    近似梯度 arctan
    """
    @staticmethod
    def forward(ctx, spsp,vth):
        ctx.save_for_backward(spsp)
        ctx.vth=vth
        output = torch.gt(spsp, vth) 
        return output.type_as(spsp)

    @staticmethod
    def backward(ctx, dy):
        spsp, = ctx.saved_tensors 
        return _back(spsp, dy,ctx.vth),None
        # frac=(3.1416 /2 * alpha)**2
        # hu = alpha / (2 * (1 + input**2 * frac))
        # return dy * hu

@torch.jit.script
def _back(spsp,dy,vth:float):
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (spsp-vth)**2 * frac))
    return dy*hu

class G_rect(torch.autograd.Function):
    """ 
    近似梯度 rectangle
    """
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth=vth
        output = torch.gt(input, vth)
        return output.type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a=1.0
        grad_input = grad_output.clone()
        hu = (abs(input - ctx.vth) < (a/2)) / a
        return grad_input * hu,None