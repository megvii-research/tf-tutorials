import os
import tensorflow as tf
import numpy as np

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
data = None
dir_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.abspath(dir_path + "/../../models/resnet18.npy")


class Resnet18:
    def __init__(self, res18_npy_path=None):
        global data

        if res18_npy_path is None:
            path = weights_path

            if os.path.exists(path):
                res18_npy_path = path

            else:
                print("Resnet18 weights were not found in the project directory!")
                exit(0)

        if data is None:
            data = np.load(res18_npy_path, encoding='latin1')
            self.data_dict = data.item()
            print("Resnet18 weights loaded")

        else:
            self.data_dict = data.item()



    def residual_block_first(self, x, channel_equal = False, strides = 2, name = 'unit'):
        with tf.variable_scope(name) as scope:
            if channel_equal:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self.conv_layer(x, name = name +'_shortcut', strides = strides)

            conv1 = self.conv_layer(x, name = name + '_conv_1', strides = strides)
            relu1 = tf.nn.relu(conv1, name = name + 'relu1')
            conv2 = self.conv_layer(relu1, name = name + '_conv_2')
            merged = conv2 + shortcut
            relu2 = tf.nn.relu(merged, name = name + 'relu2')
        return relu1, relu2

    def residual_block(self, x, name = 'unit'):
        with tf.variable_scope(name) as scope:
            shortcut = x
            conv1 = self.conv_layer(x, name = name + '_conv_1')
            relu1 = tf.nn.relu(conv1, name = name + 'relu1')
            conv2 = self.conv_layer(relu1, name = name + '_conv_2')

            merged = conv2 + shortcut
            relu2 = tf.nn.relu(merged, name = name + 'relu2')
        return relu1, relu2




    def build(self, bgr_input):

        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr_input)

        rgb = tf.concat(axis=3, values=[
            (red - vgg_mean[0])/vgg_std[0],
            (green - vgg_mean[1])/vgg_std[1],
            (blue - vgg_mean[2])/vgg_std[2],
        ])


        self.conv1_1 = self.conv_layer(rgb, "conv1")
        self.pool1 = self.avg_pool(self.conv1_1, 'pool1')

        self.middle0, self.res1 = self.residual_block(self.pool1, "conv2_1")
        self.middle1, self.res2 = self.residual_block(self.res1, "conv2_2")

        _, self.res3 = self.residual_block_first(self.res2, name = 'conv_3_1')
        _, self.res4 = self.residual_block(self.res3, 'conv_3_2')

        _, self.res5 = self.residual_block_first(self.res4, name = 'conv_4_1')
        _, self.res6 = self.residual_block(self.res5, 'conv_4_2')

        _, self.res7 = self.residual_block_first(self.res6, name = 'conv_5_1')
        _, self.res8 = self.residual_block(self.res7, 'conv_5_2')


        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom,
            ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, strides = 1):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, strides, strides, 1], padding='SAME')
            mean = self.get_mean(name)
            variance = self.get_variance(name)
            offset = self.get_beta(name)
            scale = self.get_gamma(name)
            norm = tf.nn.batch_normalization(conv, mean, variance, offset, scale, 1e-20 )
            return norm

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
        return tf.constant(self.data_dict[name][1], name = "mean")

    def get_variance(self, name):
        return tf.constant(self.data_dict[name][2], name = "variance")

    def get_gamma(self, name):
        return tf.constant(self.data_dict[name][3], name = "gamma")

    def get_beta(self, name):
        return tf.constant(self.data_dict[name][4], name = "beta")

    def get_conv_filter(self, name):
        return tf.constant(np.rollaxis(np.rollaxis(np.rollaxis(self.data_dict[name][0], 1), 2), 3), name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

