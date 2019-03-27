import os
import tensorflow as tf
import numpy as np
import inspect
import urllib.request

data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.abspath(dir_path + "/../../model/vgg16-cifar.npy")


class Model():
    def __init__(self, vgg16_npy_path=None):
        global data

        if vgg16_npy_path is None:
            path = weights_path

            if os.path.exists(path):
                vgg16_npy_path = path

            else:
                print("VGG16 weights were not found in the project directory!")
                exit(0)

        if data is None:
            data = np.load(vgg16_npy_path, encoding='latin1')
            self.data_dict = data.item()
            print("VGG16 weights loaded")

        else:
            self.data_dict = data.item()

    def build(self, inp):

        self.conv1_1 = self.conv_layer(inp, "conv1_1")
        self.drop1_1 = self.drop_layer(self.conv1_1, 0.3)
        self.conv1_2 = self.conv_layer(self.drop1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.drop2_1 = self.drop_layer(self.conv2_1, 0.4)
        self.conv2_2 = self.conv_layer(self.drop2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.drop3_1 = self.drop_layer(self.conv3_1, 0.4)
        self.conv3_2 = self.conv_layer(self.drop3_1, "conv3_2")
        self.drop3_2 = self.drop_layer(self.conv3_2, 0.4)
        self.conv3_3 = self.conv_layer(self.drop3_2, "conv3_3")

        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.drop4_1 = self.drop_layer(self.conv4_1, 0.4)
        self.conv4_2 = self.conv_layer(self.drop4_1, "conv4_2")
        self.drop4_2 = self.drop_layer(self.conv4_2, 0.4)
        self.conv4_3 = self.conv_layer(self.drop4_2, "conv4_3")

        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.drop5_1 = self.drop_layer(self.conv5_1, 0.4)
        self.conv5_2 = self.conv_layer(self.drop5_1, "conv5_2")
        self.drop5_2 = self.drop_layer(self.conv5_2, 0.4)
        self.conv5_3 = self.conv_layer(self.drop5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, "pool5")

        self.drop6 = self.drop_layer(self.pool5,0.5)
        self.fc1 = self.fc_layer(self.drop6, 'fc1', ac = True, bn = True)
        self.drop1 = self.drop_layer(self.fc1, 0.5)

        logits = self.fc_layer(self.drop1, 'fc2')

        return logits

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, stride = 1):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            mean = self.get_mean(name)
            variance = self.get_variance(name)
            offset = self.get_beta(name)
            scale = self.get_gamma(name)
            relu = tf.nn.relu(bias)
            norm = tf.nn.batch_normalization(relu, mean, variance, offset, scale, 1e-20 )
            return norm

    def drop_layer(self, bottom, rate):
        return tf.nn.dropout(bottom, rate)

    def fc_layer(self, bottom, name, ac = False, bn = False):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            if (ac):
                fc = tf.nn.relu(fc)
            if (bn):
                mean = self.get_mean(name)
                variance = self.get_variance(name)
                offset = self.get_beta(name)
                scale = self.get_gamma(name)
                fc = tf.nn.batch_normalization(fc, mean, variance, offset, scale, 1e-20 )
            return fc

    def get_mean(self, name):
        return tf.constant(self.data_dict[name][4], name = "mean")

    def get_variance(self, name):
        return tf.constant(self.data_dict[name][5], name = "variance")

    def get_gamma(self, name):
        return tf.constant(self.data_dict[name][2], name = "gamma")

    def get_beta(self, name):
        return tf.constant(self.data_dict[name][3], name = "beta")

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

