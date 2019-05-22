#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : common.py
# Purpose : configure the settings of 01-svhn
# Creation Date : 2019-02-19 10:30
# Last Modified :
# Created By : sunpeiqin
# =======================================

import os

class Config:
    '''where to write all the logging information during training(includes saved models)'''
    log_dir = './train_log'

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    exp_name = os.path.basename(log_dir)
    nr_channel = 3
    nr_epoch = 5000
    '''save the image every 10 epoch'''
    save_interval = 10
    '''show the training loss every 10 epoch'''
    show_interval = 10




config = Config()
