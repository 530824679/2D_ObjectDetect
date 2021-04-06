
import os
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, groups=1, active=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, kernel // 2, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.LeakyReLU(0.1, inplace=True) if active else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, shortcut=True, group=1, rate=0.5):
        super(Bottleneck, self).__init__()
        mid_channel = int(out_channel * rate)
        self.conv_1 = Conv(in_channel, mid_channel, 1, 1)
        self.conv_3 = Conv(mid_channel, out_channel, 3, 1, group=group)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        if self.add:
            net = self.conv_3(self.conv_1(x))
            out = x + net
        else:
            out = self.conv_3(self.conv_1(x))
        return out

class BottleneckCSP(nn.Module):
     def __init__(self, in_channel, out_channel, num, shortcut=True, group=1, rate=0.5):
         super(BottleneckCSP, self).__init__()
         mid_channel = int(out_channel * rate)
         self.conv_1 = Conv(in_channel, mid_channel, 1, 1)
         self.conv_2 = nn.Conv2d(in_channel, mid_channel, 1, 1, bias=False)
         self.conv_3 = nn.Conv2d(mid_channel, mid_channel, 1, 1, bias=False)
         self.conv_4 = Conv(out_channel, out_channel, 1, 1)
         self.bn = nn.BatchNorm2d(2 * mid_channel)  # applied to cat(conv_2, conv_3)
         self.act = nn.LeakyReLU(0.1, inplace=True)
         self.m = nn.Sequential(*[Bottleneck(mid_channel, mid_channel, shortcut, group, e=1.0) for _ in range(num)])

     def forward(self, x):
         out_1 = self.conv_3(self.m(self.conv_1(x)))
         out_2 = self.conv_2(x)
         out =  self.conv_4(self.act(self.bn(torch.cat((out_1, out_2), dim=1))))
         return out

class SPP(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=(5, 9, 13)):
        super(SPP, self).__init__()
        mid_channel = in_channel // 2
        self.conv_1 = Conv(in_channel, mid_channel, 1, 1)
        self.fusion = Conv(mid_channel * (len(kernel) + 1), out_channel, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])

    def forward(self, x):
        x = self.conv_1(x)
        return self.fusion(torch.cat([x] + [m(x) for m in self.m], 1))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Focus(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1):
        super(Focus, self).__init__()
        self.conv = Conv(in_channel * 4, out_channel, kernel, 1)

    def forward(self, x):
        split = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        out = self.conv(split, 1)
        return out

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dim = dimension

    def forward(self, x):
        return torch.cat(x, self.dim)