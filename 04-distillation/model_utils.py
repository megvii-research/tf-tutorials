#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def round_bit(x, num_bit):
    max_val = 2 ** num_bit - 1
    y = x + tf.stop_gradient((tf.floor(x * max_val + 0.5)) / max_val-x)
    return y


def quantize_w(x, num_bit):
    scale = tf.reduce_mean(tf.abs(x)) * 2
    y = x+ tf.stop_gradient((round_bit(tf.clip_by_value(x / scale, -0.5, 0.5) + 0.5, num_bit=num_bit) - 0.5) * scale - x)
    return y


def proc(x, multiplier, num_bit):
    x = tf.clip_by_value(x * multiplier, 0, 1)
    x = round_bit(x, num_bit=num_bit)
    return x


def conv2d_quantize(name, inp, stride, padding, data_format, kernel_shape, conv_filter, conv_bias, f_num_bits = None, w_num_bits = None, if_num_bits=None, is_training=False):
    if if_num_bits is None:
        if_num_bits = f_num_bits

    if w_num_bits != 0:
        quantize_conv_filter = quantize_w(tf.tanh(conv_filter), w_num_bits)

    _, _, _, oc = quantize_conv_filter.get_shape().as_list()
    conv = tf.nn.conv2d(inp, quantize_conv_filter, strides = [1, stride, stride, 1], padding=padding, data_format='NHWC')
    bias = tf.nn.bias_add(conv, conv_bias, data_format ='NHWC')
    bn = tf.layers.batch_normalization(bias, axis=3, training=is_training, center = False, scale = False)
    affine_k = tf.get_variable(name = name+'affine_k', initializer=np.array(np.ones((1, 1, 1, oc)), dtype=np.float32))
    affine_b = tf.get_variable(name = name+'affine_b', initializer=np.array(np.zeros((1, 1, 1, oc)), dtype=np.float32))
    affine = (tf.abs(affine_k) + 1.0) * bn + affine_b
    
    if f_num_bits != 0:
        out = proc(affine, 0.1, f_num_bits)
   
    return out
