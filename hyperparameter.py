# coding=UTF-8
# coder: Zxdon
# github: https://github.com/Zxdon/residual-net-tensorflow-self

# -----------------------------------------------------------------

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(flag_name='height', default_value=32, docstring='height of image')
tf.app.flags.DEFINE_integer(flag_name='width', default_value=32, docstring='width of image')
tf.app.flags.DEFINE_integer(flag_name='channel', default_value=3, docstring='channel of image')
tf.app.flags.DEFINE_integer(flag_name='padding_size', default_value=2, docstring='pad size')

tf.app.flags.DEFINE_integer(flag_name='batch_size', default_value=128, docstring='batch size')
tf.app.flags.DEFINE_integer(flag_name='epoch', default_value=10, docstring='epoch')
tf.app.flags.DEFINE_integer(flag_name='step_per_epoch', default_value=391, docstring='train steps of nn')
tf.app.flags.DEFINE_integer(flag_name='report_step', default_value=391, docstring='train steps of nn')
tf.app.flags.DEFINE_float(flag_name='init_lr', default_value=0.1, docstring='init learn rate of nn')
tf.app.flags.DEFINE_float(flag_name='lr_decay', default_value=0.1, docstring='learn rate decay')
tf.app.flags.DEFINE_integer(flag_name='first_lr_decay_step', default_value=4000, docstring='the first step for lr decay')
tf.app.flags.DEFINE_integer(flag_name='second_lr_decay_step', default_value=6000,
                            docstring='the second step for lr decay')

tf.app.flags.DEFINE_string(flag_name='log_dir', default_value='./logs', docstring='logs of tensorflow graph')
tf.app.flags.DEFINE_float(flag_name='weight_decay', default_value=0.002, docstring='weight decay for regularization')