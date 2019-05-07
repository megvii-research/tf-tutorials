#!/usr/bin/env python
# vim: ts=4 sw=4 sts=4 expandtab

import os
import argparse
import tensorflow as tf

from model import Model
from bigmodel import BigModel
from dataset import Dataset
from common import config

def get_dataset_batch(ds_name):
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset().from_generator(ds_gnr, output_types=(tf.float32, tf.int32),)
    if ds_name == 'train':
        ds = ds.shuffle(dataset.instances_per_epoch)
        ds = ds.repeat(config.nr_epoch)
    elif ds_name == 'test':
        ds = ds.repeat(config.nr_epoch // config.test_interval)
    ds = ds.batch(config.minibatch_size, drop_remainder=True)
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset

def train(args, teacher_network):
    g1 = tf.Graph()

    with g1.as_default():
        train_batch_gnr, train_set = get_dataset_batch(ds_name='train')
        test_batch_gnr, test_set = get_dataset_batch(ds_name='test')

        network = Model()
        placeholders, label_onehot, logits = network.build() ## build quantized model
        preds = tf.nn.softmax(logits)
        target_label_onehot = tf.placeholder(tf.float32, shape=(None,) + (config.nr_class,), name='target_label') ### teacher label
        correct_pred = tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                            tf.cast(tf.argmax(label_onehot, 1), dtype=tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss = tf.losses.softmax_cross_entropy(target_label_onehot, logits/args.temperature)  + loss_reg

        global_steps = tf.Variable(0, trainable=False)
        boundaries = [train_set.minibatchs_per_epoch*15, train_set.minibatchs_per_epoch*40]
        values = [0.01, 0.001, 0.0005]
        lr = tf.train.piecewise_constant(global_steps, boundaries, values)
        opt = tf.train.AdamOptimizer(lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='quantize')
        trainable_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='quantize')
        with tf.control_dependencies(update_ops):
            train = opt.minimize(loss, var_list= trainable_varlist)
        
        tf.summary.scalar('loss_regularization', loss_reg)
        tf.summary.scalar('loss_crossEntropy', loss - loss_reg)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())
        test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'test'),
                                        tf.get_default_graph())

        saver = tf.train.Saver(var_list=trainable_varlist)

    tf.set_random_seed(12345) # ensure consistent results
    global_cnt = 0
    epoch_start = 0

    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer()) # init all variables
        if args.continue_path: # load a model snapshot
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start * train_set.minibatchs_per_epoch

        ## training
        for epoch in range(epoch_start+1, config.nr_epoch+1):
            for _ in range(train_set.minibatchs_per_epoch):
                global_cnt += 1
                images, labels = sess.run(train_batch_gnr)
                target_labels = teacher_network.get_prediction(images) ## assign teacher label here
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    target_label_onehot: target_labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: True,
                }
                _, loss_v, loss_reg_v, acc_v, lr_v, summary = sess.run([train, loss, loss_reg,
                                                                        accuracy, lr, merged],
                                                                       feed_dict=feed_dict)
                if global_cnt % config.show_interval == 0:
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{},{}/{}".format(epoch, global_cnt % train_set.minibatchs_per_epoch,
                                            train_set.minibatchs_per_epoch),
                        'loss: {:.3f}'.format(loss_v),
                        'loss_reg: {:.3f}'.format(loss_reg_v),
                        'acc: {:.3f}'.format(acc_v),
                        'lr: {:.3f}'.format(lr_v),
                    )

            ## validation
            if epoch % config.test_interval == 0:
                loss_sum = 0
                acc_sum = 0
                for i in range(test_set.minibatchs_per_epoch):
                    images, labels = sess.run(test_batch_gnr)
                    target_labels = sess.run(label_onehot, feed_dict = {placeholders['label']:labels})
                    feed_dict = {
                        placeholders['data']: images,
                        placeholders['label']: labels,
                        target_label_onehot: target_labels,
                        global_steps: global_cnt,
                        placeholders['is_training']: False
                    }
                    loss_v, acc_v, summary = sess.run([loss, accuracy, merged],
                                                      feed_dict=feed_dict)
                    loss_sum += loss_v
                    acc_sum += acc_v
                test_writer.add_summary(summary, global_cnt)
                print("\n**************Validation results****************")
                print('loss_avg: {:.3f}'.format(loss_sum/test_set.minibatchs_per_epoch),
                      'accuracy_avg: {:.3f}'.format(acc_sum/test_set.minibatchs_per_epoch))
                print("************************************************\n")

            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

        print('Training is done, exit.')



