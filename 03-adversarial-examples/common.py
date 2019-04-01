#!/usr/bin/env python

import os

class Config:
    '''where to write all the logging information during training(includes saved models)'''
    log_dir = './train_log'

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    exp_name = os.path.basename(log_dir)

    nr_instances = 100 ### choose 100 images for generating adversarial examples
    minibatch_size = 1
    nr_channel = 3
    image_shape = (32, 32)
    nr_class = 10
    nr_epoch = 500 ### you may need to increase nr_epoch to 4000 or more for targeted adversarial attacks

    weight_decay = 1e-10

    show_interval = 1
    
    '''mean and standard deviation for normalizing the image input '''
    mean = 120.707
    std = 64.15
    @property
    def input_shape(self):
        return (self.minibatch_size, self.nr_channel) + self.image_shape

config = Config()
