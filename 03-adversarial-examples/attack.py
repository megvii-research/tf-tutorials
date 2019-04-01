#!/usr/bin/env python

# vim: ts=4 sw=4 sts=4 expandtab

import numpy as np
import tensorflow as tf
from common import config

mean, std = config.mean, config.std

class Attack():

    def __init__(self, model, batchsize, **kwargs):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier
    
    '''Build computation graph for generating adversarial examples'''
    def generate_graph(self, pre_noise, x, gt, target = None, **kwargs):
        noise = 10 * tf.tanh(pre_noise) 
        x_noise = x + noise                 ## add perturbation and get adversarial examples
        x_clip = tf.clip_by_value(x_noise, 0, 255) 
        x_round = x_clip + tf.stop_gradient(x_clip // 1 - x_clip) ##skip computing gradient wrt to rounded results(x_round) and only calculate the gradient wrt to x_clip 
        x_norm = (x_round - mean)/(std + 1e-7)          ## normalize the image input for the classfier
        logits = self.model.build(x_norm)               
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        if target != None:
            target_one_hot = tf.one_hot(target, config.nr_class)
        else:
            target_one_hot = tf.one_hot(gt, config.nr_class)

        loss = tf.losses.softmax_cross_entropy(target_one_hot, logits) * (-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype = tf.int32)), tf.float32))
        return  acc, loss, x_round

    '''Build a graph for evaluating the classification result of adversarial examples'''
    def evaluate(self, x, gt, **kwargs): 
        x = (x - mean)/(std + 1e-7)
        logits = self.model.build(x)
        preds = tf.nn.softmax(logits)
        gt_one_hot = tf.one_hot(gt, config.nr_class)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                             tf.cast(tf.argmax(gt_one_hot, 1), dtype = tf.int32)), tf.float32))


        return acc
