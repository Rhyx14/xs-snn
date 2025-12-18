import torch.nn as nn
import torch
import einops
# class TEBN(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super(TEBN, self).__init__()
#         self.bn = nn.BatchNorm3d(num_features)
#         self.p = nn.Parameter(torch.ones(4, 1, 1, 1, 1))

#     def forward(self, input):
#         y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
#         y = self.bn(y)
#         y = y.contiguous().transpose(1, 2)
#         y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
#         y = y * self.p
#         y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
#         return y

class TEBN(nn.Module):
    def __init__(self, inplane,frame_count,*kwds):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(inplane)
        self.p = nn.Parameter(torch.ones(frame_count, 1, 1, 1, 1))

    def forward(self, input):
        y = einops.rearrange(input,'t b c h w -> b c t h w').contiguous()
        y = self.bn(y)
        y = einops.rearrange(y,'b c t h w -> t b c h w').contiguous()
        y = y * self.p
        return y