#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : bigmodel.py
# Purpose : build a compute graph
# Creation Date : 2019-05-18 16:02
# Last Modified :
# Created By : niuyazhe
# =======================================


import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization_utils import QuantConv2d


class Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, out_f_num_bits=None, w_num_bits=None, is_train=False):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.out_f_num_bits = out_f_num_bits
        self.w_num_bits = w_num_bits
        self.is_train = is_train
        self.build()
        self.init()

    def _conv_layer(self, *args, **kwargs):
        layer = QuantConv2d(*args, **kwargs)
        return layer

    def _pool_layer(self, kernel_size, stride, padding=0, mode='MAX'):

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
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                if mode == 'FAN_IN':
                    n = i
                elif mode == 'FAN_OUT':
                    n = o
                m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                m.bias.data.zero_()

    def build(self):
        w = self.w_num_bits
        out_f = self.out_f_num_bits
        self.conv1 = self._conv_layer("conv1", self.in_channels, 16, 3, 1,
                                      1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.mp1 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv21 = self._conv_layer(
            "conv21", 16, 32, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.conv22 = self._conv_layer(
            "conv22", 32, 32, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.mp2 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv31 = self._conv_layer(
            "conv31", 32, 64, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.conv32 = self._conv_layer(
            "conv32", 64, 64, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.mp3 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv41 = self._conv_layer(
            "conv41", 64, 128, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
        self.conv42 = self._conv_layer(
            "conv42", 128, 128, 3, 1, 1, w_num_bits=w, out_f_num_bits=out_f, is_train=self.is_train)
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
    net = Model(w_num_bits=2, out_f_num_bits=1)
    inputs = torch.randn(4, 3, 32, 32)
    output = net(inputs)
    print(output.shape)


if __name__ == "__main__":
    test_model()
