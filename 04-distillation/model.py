#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib as tf_contrib
from common import config
from model_utils import conv2d_quantize


class Model():
    def __init__(self):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)

    def _conv_layer(self, name, inp, kernel_shape, stride, padding='SAME', f_num_bits=None,w_num_bits = None, if_num_bits = None, is_training=False):
        with tf.variable_scope(name) as scope:

            conv_filter = tf.get_variable(name=name+'_filter', shape=kernel_shape, initializer=self.weight_init, regularizer=self.reg)

            conv_bias = tf.get_variable(name=name+'_bias', shape=kernel_shape[-1], initializer=self.bias_init)
            x = conv2d_quantize(name + '_quantize', inp, stride, padding, data_format='NHWC',kernel_shape = kernel_shape, conv_filter = conv_filter, conv_bias = conv_bias, f_num_bits = f_num_bits, w_num_bits = w_num_bits, if_num_bits = if_num_bits, is_training = is_training )
        return x

    def _pool_layer(self, name, inp, ksize, stride, padding='SAME', mode='MAX'):
        assert mode in ['MAX', 'AVG'], 'the mode of pool must be MAX or AVG'
        if mode == 'MAX':
            x = tf.nn.max_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        elif mode == 'AVG':
            x = tf.nn.avg_pool(inp, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                               padding=padding, name=name, data_format='NHWC')
        return x

    def _fc_layer(self, name, inp, units, dropout=0.5):
        with tf.variable_scope(name) as scope:
            shape = inp.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(inp, [-1, dim]) # flatten
            if dropout > 0:
                x = tf.nn.dropout(x, keep_prob=dropout, name='dropout')
            x = tf.layers.dense(x, units, kernel_initializer=self.weight_init,
                                bias_initializer=self.bias_init, kernel_regularizer=self.reg)
        return x

    def build(self):
        w_bits = 2
        f_bits = 1
        data = tf.placeholder(tf.float32, shape=(None,)+config.image_shape+(config.nr_channel,),
                              name='data')
        label = tf.placeholder(tf.int32, shape=(None,), name='label')
        # convert the format of label to one-hot
        label_onehot = tf.one_hot(label, config.nr_class, dtype=tf.int32)
        # a setting for bn
        is_training = tf.placeholder(tf.bool, name='is_training')

        # conv1
        x = self._conv_layer(name='quantized_conv1', inp=data,
                             kernel_shape=[3, 3, config.nr_channel, 16], stride=1, f_num_bits = f_bits, w_num_bits = w_bits, is_training=is_training) # Nx32x32x32
        x = self._pool_layer(name='quantized_pool1', inp=x, ksize=2, stride=2, mode='MAX') # Nx16x16x16

        # conv2
        x = self._conv_layer(name='quantized_conv21', inp=x, kernel_shape=[3, 3, 16, 32],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits,is_training=is_training)
        x = self._conv_layer(name='quantized_conv22', inp=x, kernel_shape=[3, 3, 32, 32],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits, is_training=is_training)
        x = self._pool_layer(name='quantized_pool2', inp=x, ksize=2, stride=2, mode='MAX') # Nx8x8x32

        # conv3
        x = self._conv_layer(name='quantized_conv31', inp=x, kernel_shape=[3, 3, 32, 64],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits,is_training=is_training)
        x = self._conv_layer(name='quantized_conv32', inp=x, kernel_shape=[3, 3, 64, 64],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits,is_training=is_training)
        x = self._pool_layer(name='quantized_pool3', inp=x, ksize=2, stride=2, mode='MAX') # Nx4x4x64

        # conv4
        
        x = self._conv_layer(name='quantized_conv41', inp=x, kernel_shape=[3, 3, 64, 128],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits,is_training=is_training)
        x = self._conv_layer(name='quantized_conv42', inp=x, kernel_shape=[3, 3, 128, 128],
                             stride=1, f_num_bits = f_bits, w_num_bits = w_bits,is_training=is_training)
        x = self._pool_layer(name='quantized_pool4', inp=x, ksize=4, stride=4, mode='AVG') # Nx1x1x128
        
        # fc1
        logits = self._fc_layer(name='quantized_fc1', inp=x, units=config.nr_class, dropout=0)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
        }
        return placeholders, label_onehot, logits
