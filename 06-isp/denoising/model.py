#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tf_contrib
from common import config


class Model():
    def __init__(self, depth = 17):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.depth = depth
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME',is_training=False, include_bn = True, include_relu = True ):
        with tf.variable_scope(name) as scope:
            conv_filter = tf.get_variable(name='filter', shape=kernel_shape,
                                          initializer=self.weight_init, regularizer=self.reg)
            conv_bias = tf.get_variable(name='bias', shape=kernel_shape[-1],
                                        initializer=self.bias_init)
            x = tf.nn.conv2d(inp, conv_filter, strides=[1, stride, stride, 1],
                             padding=padding, data_format='NHWC')
            x = tf.nn.bias_add(x, conv_bias, data_format='NHWC')
            if include_bn == True:
                x = tf.layers.batch_normalization(x, axis=3, training=is_training)
            if include_relu == True:
                x = tf.nn.relu(x)
        return x


    def build(self):
        data = tf.placeholder(tf.float32, shape=(None,)+(None,None, )+(config.nr_channel,),
                              name='data')
        # a setting for bn
        is_training = tf.placeholder(tf.bool, name='is_training')


        x = self._conv_layer(name='conv0', inp=data,
                             kernel_shape=[3, 3, config.nr_channel, 64], stride=1,
                             is_training=is_training, include_bn = False) # Nx32x32x32


        for i in range(self.depth - 2):
            x = self._conv_layer(name = 'conv'+ str(i+1), inp = x, kernel_shape = [3, 3, 64, 64 ], stride = 1, is_training = is_training )


        x = self._conv_layer(name='conv' + str(self.depth), inp = x, kernel_shape=[3, 3, 64, config.nr_channel],
                             stride=1, is_training=is_training, include_bn = False, include_relu = False)

        output = data - x ## residual learning

        placeholders = {
            'data': data,
            'is_training': is_training,
        }
        return placeholders, output
