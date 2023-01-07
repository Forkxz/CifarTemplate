import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """ Standard convolution """
    def __init__(self, in_planes, planes, kernel_size=1, stride=1, padding=None, bias=False, group=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size, stride, autopad(kernel_size, padding), groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))