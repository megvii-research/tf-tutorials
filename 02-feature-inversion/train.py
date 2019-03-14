#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name : train.py
# Purpose : run a session for training
# Creation Date : 2019-03-11
# Last Modified :
# Created By : wangtian
# =======================================

import os
import argparse
import tensorflow as tf

from common import config
import custom_vgg16_bn as vgg16
import cv2
import numpy as np

'''use conv3_1 to generate representation'''
FEATURE_LAYERS = ['conv3_1']
image_shape = (1, 224, 224, 3)

def get_feature_loss(noise,source):
    with tf.name_scope('get_feature_loss'):
        feature_loss = [get_l2_loss_for_layer(noise, source, layer)for layer in FEATURE_LAYERS]
    return tf.reduce_mean(tf.convert_to_tensor(feature_loss))

def get_l2_loss_for_layer(noise, source, layer):
    noise_layer = getattr(noise,layer)
    source_layer = getattr(source, layer)
    l2_loss = tf.reduce_mean((source_layer-noise_layer) **2)
    return  l2_loss

def output_img(session, x, save=False, out_path=None):
    shape = image_shape
    img = np.clip(session.run(x),0, 1) * 255
    img = img.astype('uint8')
    if save:
        cv2.imwrite(out_path, (np.reshape(img, shape[1:])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    '''training a image from noise that resemble the target'''
    pre_noise = tf.Variable(tf.random_uniform(image_shape, -3, 3 ))
    noise = tf.Variable(tf.nn.sigmoid(pre_noise))

    '''load target image, notice that the pixel value has to be normalized to [0,1]'''
    image = cv2.imread('../images/face.jpg')
    image = cv2.resize(image, image_shape[1:3])
    image = image.reshape(image_shape)
    image = (image/255).astype('float32')

    '''get representation of the target image, which the noise image will approximate'''
    with tf.name_scope('vgg_src'):
        image_model = vgg16.Model()
        image_model.build(image)

    '''get representation of the noise image'''
    with tf.name_scope('vgg_noise'):
        noise_model = vgg16.Model()
        noise_model.build(noise)

    '''compute representation difference between noise feature and target feature'''
    with tf.name_scope('loss'):
        loss = get_feature_loss(noise_model, image_model)

    total_loss = loss

    global_steps = tf.Variable(0, trainable=False)
    lr = 1e-3

    with tf.name_scope('update_image'):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss, [noise])
        update_image = opt.apply_gradients(grads)

    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())

    ''' create a session '''
    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # init all variables
        if args.continue_path: # load a model snapshot
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start

        ## training
        for epoch in range(epoch_start+1, config.nr_epoch+1):
            global_cnt += 1
            _, loss, summary =  sess.run([update_image, total_loss, merged],
                                        feed_dict ={ global_steps: global_cnt} )

            if global_cnt % config.show_interval == 0:
                train_writer.add_summary(summary, global_cnt)
                print(
                    "e:{}".format(epoch),'loss: {:.5f}'.format(loss),
                )

            '''save the trained noise image every 10 epoch, check whether it resembles the target image'''
            if global_cnt % config.save_interval == 0 and global_cnt >0 :
                out_dir = os.path.dirname(os.path.realpath(__file__)) + '/./output'

                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)
                out_dir = out_dir +'/{}.png'.format(global_cnt)
                output_img(sess, noise, save=True, out_path = out_dir)


            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

        print('Training is done, exit.')




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)

# vim: ts=4 sw=4 sts=4 expandtab
