#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : model.py
# Purpose : build a compute graph
# Creation Date : 2019-03-23 14:02
# Last Modified :
# Created By : niuyazhe
# =======================================


import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from generate_gauss import gauss2D


class LpPool2d(nn.Module):
    def __init__(self, p, kernel_size, stride, padding=0):
        super(LpPool2d, self).__init__()
        self.p = p
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        if isinstance(p, numbers.Integral) and p > 0:
            self.gauss_kernel = gauss2D((kernel_size, kernel_size))
            self.gauss_kernel = torch.from_numpy(self.gauss_kernel).float()
            if torch.cuda.is_available():
                self.gauss_kernel = self.gauss_kernel.cuda()
        else:
            raise ValueError

    def forward(self, x):
        x_p = torch.pow(x, self.p)
        weight = self.gauss_kernel.unsqueeze(0).unsqueeze(0).repeat(x_p.size()[1], 1, 1, 1)
        output = F.conv2d(x_p, weight, stride=self.stride, padding=self.padding, groups=x_p.size()[1])
        output_1p = torch.pow(output, 1./self.p)
        return output_1p


class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, pool='normal'):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pool = pool
        if self.pool != 'normal' and not isinstance(self.pool, numbers.Integral):
            raise ValueError
        self.build()
        #self.init()

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU(), use_bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            layers.append(activation)
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel_size, stride, padding=0, mode='MAX'):
        if isinstance(self.pool, numbers.Integral):
            return LpPool2d(self.pool, kernel_size, stride, padding)

        mode_list = ['MAX', 'AVG']
        assert(mode in mode_list or isinstance(mode, numbers.Integral))
        if mode == 'MAX':
            return nn.MaxPool2d(kernel_size, stride, padding)
        elif mode == 'AVG':
            return nn.AvgPool2d(kernel_size, stride, padding)

    def _fc_layer(self, in_channels, out_channels, dropout=0.5):
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        if dropout != 0:
            layers.append(nn.Dropout2d(p=dropout))
        return nn.Sequential(*layers)

    def init(self, scale_factor=1.0, mode='FAN_IN'):
        mode_list = ['FAN_IN']
        assert(mode in mode_list)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'FAN_IN':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                elif mode == 'FAN_OUT':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def build(self):
        self.conv1 = self._conv_layer(self.in_channels, 16, 3, 1, 1)
        self.mp1 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv21 = self._conv_layer(16, 32, 3, 1, 1)
        self.conv22 = self._conv_layer(32, 32, 3, 1, 1)
        self.mp2 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv31 = self._conv_layer(32, 64, 3, 1, 1)
        self.conv32 = self._conv_layer(64, 64, 3, 1, 1)
        self.mp3 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv41 = self._conv_layer(64, 128, 3, 1, 1)
        self.conv42 = self._conv_layer(128, 128, 3, 1, 1)
        self.ap4 = self._pool_layer(kernel_size=4, stride=4, mode='AVG')

        self.fc1 = self._fc_layer(128, self.num_classes, dropout=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.mp3(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.ap4(x)

        x = self.fc1(x.view(-1, 128))

        return x


def test_model():
    net = Model()
    inputs = torch.randn(4, 3, 32, 32)
    output = net(inputs)
    print(output.shape)


def test_LpPool2d():
    pool = LpPool2d(p=2, kernel_size=2, stride=2, padding=0)
    pool.cuda()
    inputs = torch.randn(4, 3, 32, 32).cuda()
    output = pool(inputs)
    print(output.shape)


if __name__ == "__main__":
    #test_model()
    test_LpPool2d()
