import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers


def round_bits(x, num_bits):
    assert(isinstance(x, torch.Tensor))
    assert(isinstance(num_bits, numbers.Integral))
    max_val = 2 ** num_bits - 1
    round_term = torch.floor(x*max_val + 0.5) / max_val - x
    round_term = round_term.detach()
    y = x + round_term
    return y


def quantize_weight(x, num_bits):
    assert(isinstance(x, torch.Tensor))
    assert(isinstance(num_bits, numbers.Integral))
    scale = torch.abs(x).mean() * 2
    round_term = round_bits(torch.clamp(x/scale, -0.5, 0.5)+0.5, num_bits) - 0.5
    round_term = round_term*scale - x
    round_term = round_term.detach()
    y = x + round_term
    return x


def proc(x, multiplier, num_bits):
    x = torch.clamp(x*multiplier, 0, 1)
    x = round_bits(x, num_bits)
    return x


class QuantConv2d(nn.Module):
    def __init__(self, name, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 out_f_num_bits=None, w_num_bits=None, is_train=False, proc_multiplier=0.1):
        super(QuantConv2d, self).__init__()
        self.name = name
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.w_num_bits = w_num_bits
        self.out_f_num_bits = out_f_num_bits
        self.is_train = is_train
        self.proc_multiplier = proc_multiplier

        if isinstance(kernel_size, numbers.Integral):
            kernel_size = (kernel_size, kernel_size)
        self.weight = torch.zeros(
            out_channels, in_channels, *kernel_size).float()
        self.weight_init_(self.weight)
        self.bias = torch.zeros(out_channels)
        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(self.bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.affine_k = nn.Parameter(torch.ones(out_channels, 1, 1))
        self.affine_b = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def weight_init_(self, x, scale_factor=1.0, mode='FAN_IN'):
        mode_list = ['FAN_IN', 'FAN_OUT']
        assert(mode in mode_list)
        if mode == 'FAN_IN':
            o, i, h, w = x.data.shape
            n = i*h*w
            x.normal_(0, np.sqrt(scale_factor / n))
        elif mode == 'FAN_OUT':
            o, i, h, w = x.data.shape
            n = o*h*w
            x.normal_(0, np.sqrt(scale_factor / n))

    def forward(self, x):
        self.weight = quantize_weight(self.weight, num_bits=self.w_num_bits)
        x = F.conv2d(x, self.weight, self.bias,
                     stride=self.stride, padding=self.padding)
        x = self.bn(x)
        x = (torch.abs(self.affine_k)+1.0) * x + self.affine_b
        if self.out_f_num_bits != 0:
            x = proc(x, self.proc_multiplier, self.out_f_num_bits)
        return x


def test_quantize_conv2d():
    layer = QuantConv2d("quant_test", 3, 16, 3, 1, 0,
                        out_f_num_bits=1, w_num_bits=2)
    inputs = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    inputs = round_bits(inputs, 1)
    output = layer(inputs)
    print(output.shape)
    print(output[0, 0, 0])


if __name__ == "__main__":
    test_quantize_conv2d()
