#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : vgg.py
# Purpose : build a compute graph
# Creation Date : 2019-04-20 20:02
# Last Modified :
# Created By : niuyazhe
# =======================================


import math
import torch
import torch.nn as nn


class Vgg16BnCifar(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Vgg16BnCifar, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.build()
        self.init()

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU(), use_bn=True, dropout=0):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding, bias=bias))
        if activation is not None:
            layers.append(activation)
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel_size, stride, padding=0, mode='MAX'):

        mode_list = ['MAX', 'AVG']
        assert(mode in mode_list)
        if mode == 'MAX':
            return nn.MaxPool2d(kernel_size, stride, padding)
        elif mode == 'AVG':
            return nn.AvgPool2d(kernel_size, stride, padding)

    def _fc_layer(self, in_channels, out_channels, activation=None, dropout=0, use_bn=False):
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        if activation is not None:
            layers.append(activation)
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)

    def init(self, scale_factor=2.0, mode='FAN_OUT'):
        mode_list = ['FAN_IN', 'FAN_OUT']
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def build(self):
        self.conv11 = self._conv_layer(
            self.in_channels, 64, 3, 1, 1, dropout=0.3)
        self.conv12 = self._conv_layer(64, 64, 3, 1, 1)
        self.mp1 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv21 = self._conv_layer(64, 128, 3, 1, 1, dropout=0.4)
        self.conv22 = self._conv_layer(128, 128, 3, 1, 1)
        self.mp2 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv31 = self._conv_layer(128, 256, 3, 1, 1, dropout=0.4)
        self.conv32 = self._conv_layer(256, 256, 3, 1, 1, dropout=0.4)
        self.conv33 = self._conv_layer(256, 256, 3, 1, 1)
        self.mp3 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv41 = self._conv_layer(256, 512, 3, 1, 1, dropout=0.4)
        self.conv42 = self._conv_layer(512, 512, 3, 1, 1, dropout=0.4)
        self.conv43 = self._conv_layer(512, 512, 3, 1, 1)
        self.mp4 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.conv51 = self._conv_layer(512, 512, 3, 1, 1, dropout=0.4)
        self.conv52 = self._conv_layer(512, 512, 3, 1, 1, dropout=0.4)
        self.conv53 = self._conv_layer(512, 512, 3, 1, 1)
        self.mp5 = self._pool_layer(kernel_size=2, stride=2, mode='MAX')

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = self._fc_layer(
            512, 512, activation=nn.ReLU(), dropout=0.5, use_bn=True)
        self.fc2 = self._fc_layer(512, self.num_classes)

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.mp1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.mp3(x)
        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.mp4(x)
        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)
        x = self.mp5(x)

        x = x.view(-1, 512)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


def test_model():
    net = Vgg16BnCifar(num_classes=10)
    inputs = torch.randn(4, 3, 32, 32)
    output = net(inputs)
    print(output.shape)


if __name__ == "__main__":
    test_model()
