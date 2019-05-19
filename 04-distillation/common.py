#!/usr/bin/env python


import os

class Config:
    '''where to write all the logging information during training(includes saved models)'''
    log_dir = './train_log'

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')
    checkpoint_path = '../baseline_full/train_log/models'
    exp_name = os.path.basename(log_dir)

    minibatch_size = 256
    nr_channel = 3
    image_shape = (32, 32)
    nr_class = 10
    nr_epoch = 60
    weight_decay = 1e-10
    show_interval = 100
    snapshot_interval = 2
    test_interval = 1
    data_number = 73257
    use_extra_data = False

    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape

config = Config()
