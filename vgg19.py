import os
import sys

import numpy as np
import tensorflow as tf

#VGG_MEAN = [103.939, 116.779, 123.68]

# https://github.com/machrisaa/tensorflow-vgg

class Vgg16:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg19.npy")
            print(path)
            vgg16_npy_path = path

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")
        

    def build(self, input, train=False):

        self.conv1_1 = self._conv_layer(input, 1, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, 1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, 1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, 1,"conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, 1, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, 1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, 1, "conv3_3")
        self.conv3_4 = self._conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self._max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, 1, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, 1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, 1, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.s_max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, 1, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, 2, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, 3, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        #self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        #self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        #self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.s_max_pool(self.conv5_4, 'pool5')


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def s_max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                              padding='SAME', name=name)


    def _conv_layer(self, bottom, rate, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name)
            #conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv = tf.nn.atrous_conv2d(bottom, filt, rate, padding = 'SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
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

    def get_conv_filter(self, name):

        #W_regul = lambda x: self.L2(x)

        #return tf.get_variable(name="filter",
        #                       initializer=self.data_dict[name][0],
        #                       trainable=True,
        #                       regularizer=W_regul)
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")

    def L2(self, tensor, wd=0.001):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')
