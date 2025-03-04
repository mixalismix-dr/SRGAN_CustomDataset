import torch
import torch.nn as nn
import torch.nn.functional as F


class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=(kernel_size) // 2, bias=True)

        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))

        for p in self.parameters():
            p.requires_grad = True


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN=False, act=None, stride=1, bias=True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels=in_channel, out_channels=out_channel,
                       kernel_size=kernel_size, stride=stride, padding=(kernel_size) // 2, bias=True))

        if BN:
            m.append(nn.BatchNorm2d(num_features=out_channel))

        if act is not None:
            m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, act=nn.PReLU()):
        super(ResBlock, self).__init__()
        self.conv1 = conv(channels, channels, kernel_size, BN=True, act=act)
        self.conv2 = conv(channels, channels, kernel_size, BN=True, act=None)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out


class Upsampler(nn.Module):
    def __init__(self, channel=64, kernel_size=3, scale=2, act=nn.PReLU()):
        super(Upsampler, self).__init__()
        self.conv = conv(channel, channel * (scale ** 2), kernel_size, BN=False, act=act)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        out = self.pixel_shuffle(self.conv(x))
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()
        m = []

        self.conv = conv(in_channels, out_channels, kernel_size, BN=False, act=act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))

        m.append(conv(out_channels, out_channels, kernel_size, BN=True, act=None))

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res

        return out


class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act=nn.LeakyReLU(inplace=True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN=True, act=act))
        m.append(conv(out_feats, out_feats, kernel_size, BN=True, act=act, stride=2))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        out = self.body(x)
        return out


# New: Edge and Mask Processing Layers
class EdgeMaskBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act=nn.PReLU()):
        super(EdgeMaskBlock, self).__init__()
        # Edge extraction layers
        self.edge_conv1 = conv(in_channels, out_channels, kernel_size, BN=False, act=act)
        self.edge_conv2 = conv(out_channels, out_channels, kernel_size, BN=False, act=act)

        # Mask generation layers
        self.mask_conv1 = conv(out_channels, out_channels, kernel_size, BN=False, act=act)
        self.mask_conv2 = conv(out_channels, 1, kernel_size, BN=False, act=nn.Sigmoid())  # Mask has 1 channel

        # Adjust the number of channels in x_edge to match x_base (3 channels)
        self.edge_final_conv = conv(out_channels, 3, kernel_size, BN=False, act=None)

    def forward(self, x):
        # Edge extraction
        x_edge = self.edge_conv1(x)
        x_edge = self.edge_conv2(x_edge)

        # Mask generation
        mask = self.mask_conv1(x_edge)
        mask = self.mask_conv2(mask)  # Mask values in [0, 1]

        # Adjust the number of channels in x_edge to match x_base (3 channels)
        x_edge = self.edge_final_conv(x_edge)

        return x_edge, mask