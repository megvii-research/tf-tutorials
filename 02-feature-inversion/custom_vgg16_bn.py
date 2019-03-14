import os
import tensorflow as tf
import numpy as np

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.abspath(dir_path + "/../models/vgg16_onnx.npy")


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

    def build(self, bgr_input):
        '''notice that opencv load image with bgr order, but the pretrained model is designed for rgb'''
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr_input)

        rgb = tf.concat(axis=3, values=[
            (red - vgg_mean[0])/vgg_std[0],
            (green - vgg_mean[1])/vgg_std[1],
            (blue - vgg_mean[2])/vgg_std[2],
        ])


        self.conv1_1 = self.conv_layer(rgb, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")

        self.pool3 = self.avg_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")

        self.pool4 = self.avg_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

        self.pool5 = self.avg_pool(self.conv5_3, 'pool5')
        self.fc6 = self.fc_layer(self.pool5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.fc8 = self.fc_layer(self.fc7, 'fc8')
        self.data_dict = None

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
            norm = tf.nn.batch_normalization(bias, mean, variance, offset, scale, 1e-20 )
            relu = tf.nn.relu(norm)
            return relu

    def fc_layer(self, bottom, name):
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
        return tf.constant(np.rollaxis(np.rollaxis(np.rollaxis(self.data_dict[name][0], 1), 2), 3), name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(np.rollaxis(self.data_dict[name][0], 1), name="weights")

