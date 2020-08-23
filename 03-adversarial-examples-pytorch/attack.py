#!/usr/bin/env python

# vim: ts=4 sw=4 sts=4 expandtab

import torch
import torch.nn as nn


class AttackModel(nn.Module):

    def __init__(self, model, mean, std, epsilon=1e-7):
        super(AttackModel, self).__init__()
        self.model = model
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def forward(self, pre_noise, x):
        noise = 10 * torch.tanh(pre_noise)
        x_noise = x + noise
        x_clip = x_noise.clamp(0, 255)
        round_term = (x_clip//1 - x_clip).detach()
        x_round = x_clip + round_term
        x_norm = (x_round - self.mean) / (self.std + self.epsilon)
        x = self.model(x_norm)
        return x, x_round

    def evaluate(self, x):
        x = (x - self.mean)/(self.std + self.epsilon)
        x = self.model(x)
        return x
