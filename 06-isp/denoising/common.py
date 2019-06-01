#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

class Config:
    '''where to write all the logging information during training(including saved models)'''
    log_dir = './train_log'

    patch_size = 50
    stride = 10

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    exp_name = os.path.basename(log_dir)

    minibatch_size = 128
    nr_channel = 3
    nr_epoch = 160
    weight_decay = 1e-5
    show_interval = 2
    snapshot_interval = 2
    test_interval = 1
    sigma = 50
    use_extra_data = True
    scale = [1, 0.9, 0.8, 0.7]
    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape

config = Config()
