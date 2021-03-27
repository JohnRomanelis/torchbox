import torch
import torch.nn as nn
from functools import partial


def get_basic_conv_block(conv_type, norm_type, activation):
    return partial(BasicConvolutionBlock, conv_type=conv_type, norm_type=norm_type, activation=activation)

def get_residual_block(conv_type, norm_type, activation):
    return partial(ResidualBlock, conv_type=conv_type, norm_type=norm_type, activation=activation)

class BasicConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_type, norm_type, activation, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            conv_type(in_channels, out_channels, **kwargs),
            norm_type(out_channels), 
            activation()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
        Residual Layer has two roots.
        1. net: conv (sith stride) -> batchnorm -> relu -> conv(stride=1) -> batchnorm
        2. downsample : if stride : conv(with stride) -> batchnorm

        result = activation(net, downsample)

    """
    def __init__(self, in_channels, out_channels, conv_type, norm_type, activation, **kwargs):
        super().__init__()

        # getting the stride (will return none if stride is not in kwargs)
        stride = kwargs.pop('stride', None)
        if stride is None:
            stride=1

        self.net = nn.Sequential(
            conv_type(in_channels, out_channels, stride=stride,  **kwargs),
            norm_type(out_channels),
            activation(), 
            conv_type(out_channels, out_channels, stride=1, **kwargs),
            norm_type(out_channels)
        )        


        self.downsample = nn.Sequential() if (in_channels==out_channels and stride==1) else \
            nn.Sequential(
                conv_type(in_channels, out_channels, stride=stride, **kwargs),
                norm_type(out_channels)
            )

        self.activation = activation()

    def forward(self, x):
        out  = self.activation(self.net(x) + self.downsample(x))
        return out





