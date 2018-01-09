# coding=UTF-8
# coder: Zxdon
# github: https://github.com/Zxdon/residual-net-tensorflow-self

# -----------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from hyperparameter import *
import numpy as np
import tensorflow as tf

EPSILON = 0.001


class Resnet(object):
    """
    reference: Deep Residual Learning for Image Recognition
    layer num: 1 + 2n + 2n + 2n + 1 = 6n+2
    argv:
        input_image: [None, H, W, C]
        labels: [None]
        block_num: n
        is_bottle_neck: True/False
        is_train: True/False
    """
    def __init__(self, _block_num=5, _is_bottle_neck=False, _is_train=True):
        #self.input_image = _input_image
        #self.labels = _labels
        self.block_num = _block_num
        self.is_bottle_neck = _is_bottle_neck
        self.is_train = _is_train

    def activation_summary(self, x):
        """
        :param x: A Tensor shape = [None, H, W, C]
        :return: add summary
        """
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name+'/activation', x)
        tf.summary.scalar(tensor_name+'/spasity', tf.nn.zero_fraction(x))

    def output_layer(self, x, num_labels):
        """
        the last layer of nn
        :param x: shape = [None, H, W, C]
        :param num_labels: output layer label num
        :return: output shape = [None, n]
        """
        input_dim = x.get_shape().as_list()[-1]
        fc_w = tf.get_variable(name='fc_weight', shape=[input_dim, num_labels],
                               initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                               regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay))
        fc_b = tf.get_variable(name='fc_bais', shape=[num_labels],
                               initializer=tf.zeros_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay))
        return tf.nn.xw_plus_b(x, fc_w, fc_b)

    def batch_normalization(self, x):
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2])
        beta = tf.get_variable('beta', [x.get_shape().as_list()[-1]], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        gamma = tf.get_variable('gamma', [x.get_shape().as_list()[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, EPSILON)

    def bn_relu_conv(self, input_layer, output_channel, stride=1, ksize=3):
        """
        :param input_layer: shape = [None, H, W, C]
        :param output_channel: output channel after bn, relu and conv
        :param stride: the stride of conv
        :param ksize: the conv kernel size default = 3
        :return: shape = [None, H`, W`, output_channel]
        """

        bn_layer = self.batch_normalization(input_layer)
        relu_layer = tf.nn.relu(bn_layer)

        filter = tf.get_variable(name='conv', shape=[ksize, ksize, input_layer.get_shape()[-1], output_channel],
                                 initializer=tf.contrib.layers.xavier_initializer())
        return tf.nn.conv2d(input=relu_layer, filter=filter, strides=[1, stride, stride, 1], padding='SAME')

    def conv_bn_relu(self, input_layer, output_channel, stride=1, ksize=3):
        """
        :param input_layer: shape = [None, H, W, C]
        :param output_channel: output channel after conv, bn, relu
        :param stride: the stride of conv
        :param ksize: the conv kernel size default = 3
        :return: [None, H`, W`, output_channel]
        """
        filter = tf.get_variable(name='conv', shape=[ksize, ksize, input_layer.get_shape().as_list()[-1], output_channel],
                                 initializer=tf.contrib.layers.xavier_initializer())
        conv_layer = tf.nn.conv2d(input=input_layer, filter=filter, strides=[1, stride, stride, 1], padding='SAME')

        bn_layer = self.batch_normalization(conv_layer)

        return tf.nn.relu(bn_layer)

    def residual_block(self, input_layer, output_dim, is_first=False):
        """
        residual block
        :param input_layer: shape = [None, H, W, C]
        :param output_dim: if output_dim = 2*input_dim then stride = 2
        :param is_first: if it is the first block of whole net, only need conv the layer
        :return: output shape = [None, H`, W`, output_dim]
        """
        input_dim = input_layer.get_shape().as_list()[-1]

        if input_dim*2 == output_dim:
            stride = 2
            need_pad = True
        else:
            stride = 1
            need_pad = False

        with tf.variable_scope('first_conv'):
            if is_first:
                filter = tf.get_variable(name='conv', shape=[3, 3, input_dim, output_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv1 = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv(input_layer, output_dim, stride)

        with tf.variable_scope('second_conv'):
            conv2 = self.bn_relu_conv(conv1, output_dim)

        if need_pad:
            pooling_ontput = tf.nn.avg_pool(value=input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            padding_output = tf.pad(pooling_ontput, paddings=[[0, 0], [0, 0], [0, 0], [input_dim//2, input_dim//2]])
        else:
            padding_output = input_layer

        return padding_output + conv2

    def bottleneck_residual_block(self, input_layer, output_dim):
        """
        bottleneck residual block
        :param input_layer: shape = [None, H, W, C]
        :param output_dim: if output_dim = 2*input_dim then stride = 2
        :param is_first: if it is the first block of whole net, only need conv the layer
        :return: output shape = [None, H`, W`, output_dim]
        """
        input_dim = input_layer.get_shape().as_list()[-1]

        with tf.variable_scope('conv1'):
            conv1 = self.bn_relu_conv(input_layer, input_dim, ksize=1)
        with tf.variable_scope('conv2'):
            conv2 = self.bn_relu_conv(conv1, input_dim)
        with tf.variable_scope('conv3'):
            conv3 = self.bn_relu_conv(conv2, output_dim, ksize=1)

        if input_dim != output_dim:
            filter = tf.get_variable('conv', shape=[1, 1, input_dim, output_dim],
                                     initializer=tf.contrib.layers.xavier_initializer())
            output_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            output_layer = input_layer

        return output_layer + conv3

    def inference(self, input_layer, num_labels, n, is_use_bottlenec=False, reuse=False):
        """
        the whole nn structure constructed in this method
        :param input_layer: shape = [None, W, H, C]
        :param num_labels: number of labels
        :param n: block num of the nn
        :param is_use_bottlenec: whether use bottleneck in nn
        :param reuse: if train: reuse = False if valid or test: reuse = True
        :return: return output of the nn shape = [None, num_labels]
        """
        layers = []
        conv_output_channel = [16, 16, 32, 64]

        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = self.conv_bn_relu(input_layer, conv_output_channel[0])
            self.activation_summary(conv0)
            layers.append(conv0)

        if not is_use_bottlenec:
            for i in range(n):
                with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                    if i == 0:
                        conv1 = self.residual_block(layers[-1], conv_output_channel[1], is_first=True)
                    else:
                        conv1 = self.residual_block(layers[-1], conv_output_channel[1])
                    self.activation_summary(conv1)
                layers.append(conv1)

            for i in range(n):
                with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                    conv2 = self.residual_block(layers[-1], conv_output_channel[2])
                    self.activation_summary(conv2)
                layers.append(conv2)

            for i in range(n):
                with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                    conv3 = self.residual_block(layers[-1], conv_output_channel[3])
                self.activation_summary(conv3)
                layers.append(conv3)
        else:
            for i in range(n):
                with tf.variable_scope('conv1_%d' % i, reuse=reuse):
                    conv1 = self.bottleneck_residual_block(layers[-1], conv_output_channel[1])
                self.activation_summary(conv1)
                layers.append(conv1)

            for i in range(n):
                with tf.variable_scope('conv2_%d' % i, reuse=reuse):
                    conv2 = self.bottleneck_residual_block(layers[-1], conv_output_channel[2])
                self.activation_summary(conv2)
                layers.append(conv2)

            for i in range(n):
                with tf.variable_scope('conv3_%d' % i, reuse=reuse):
                    conv3 = self.bottleneck_residual_block(layers[-1], conv_output_channel[3])
                self.activation_summary(conv3)
                layers.append(conv3)

        with tf.variable_scope('fc', reuse=reuse):
            bn_layer = self.batch_normalization(layers[-1])
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            output = self.output_layer(global_pool, num_labels)
            self.activation_summary(output)
            layers.append(output)

        return layers[-1]

    def loss(self, x, labels):
        """
        caculate the loss of output
        :param x: output of the last layer
        :param labels: real labels
        :return: model loss
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def accuracy(self, x, labels):
        """
        caculate accuracy of nn
        :param x: output of the last layer
        :param labels: real labels
        :return: accuracy
        """
        softmax = tf.nn.softmax(x)
        equal = tf.equal(tf.argmax(tf.nn.softmax(softmax), 1), tf.cast(labels, dtype=tf.int64))
        accuracy = tf.reduce_mean(tf.cast(equal, dtype=tf.float32))

        return accuracy

    def top_k_error(self, x, labels, k=1):
        '''
        Calculate the top-k error
        :param x: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int
        :return: tensor with shape [1]
        '''
        batch_size = x.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(x, labels, k))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def tensor_flow_graph(self):
        input_image = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
        output = self.inference(input_image, 10, self.block_num, self.is_bottle_neck)
        print(output.get_shape().as_list())
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(output))
            accuracy, loss = sess.run([self.accuracy(output, tf.argmax(output, 1)), self.loss(output, tf.argmax(output, 1))])
            print('loss: %f, accuracy: %f' % (loss, accuracy))
            tf.summary.FileWriter(FLAGS.log_dir, sess.graph)


# model = Resnet(_block_num=5, _is_bottle_neck=True, _is_train=False)
# model.tensor_flow_graph()


# def main(_):
#     with tf.variable_scope('mul0'):
#         x = tf.get_variable('x', [3,4], dtype=tf.float32, initializer=tf.random_uniform_initializer(-1, 1))
#         y = tf.get_variable('y', [4,3], dtype=tf.float32, initializer=tf.random_uniform_initializer(-1, 1))
#
#     product = tf.matmul(x, y)
#
#     init = tf.global_variables_initializer()
#
#     with tf.Session() as sess:
#         sess.run(init)
#         print(sess.run(x))
#         print(sess.run(y))
#         print(sess.run(product))
#
#
# if __name__ == '__main__':
#     tf.app.run()