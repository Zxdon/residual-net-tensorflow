# coding=UTF-8
# coder: Zxdon
# github: https://github.com/Zxdon/residual-net-tensorflow-self

# -----------------------------------------------------------------

from residual_net import *
from hyperparameter import *
from data_util import *


class Train(object):
    '''class for train model'''
    def __init__(self):
        self.create_placeholder()

    def create_placeholder(self):
        self.input_image = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.height, FLAGS.width, FLAGS.channel])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None])
        self.lr = tf.placeholder(dtype=tf.float32, shape=[])

    def train(self):

        global_step = tf.Variable(0, trainable=False)

        model = Resnet()

        # data preparation
        train_data, train_label = prepare_train_data(FLAGS.padding_size)
        train_image_iter = train_data_iter(train_data, train_label)
        validate_data, validate_label = prepare_validate_data()

        output = model.inference(self.input_image, 10, model.block_num, model.is_bottle_neck, reuse=False)

        # top_k_error = model.top_k_error(output, self.labels)
        accuracy = model.accuracy(output, self.labels)
        loss = model.loss(output, self.labels)
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + regu_losses)

        loss_scalar = tf.summary.scalar(name='loss', tensor=total_loss)
        accuracy_scalar = tf.summary.scalar(name='accuracy', tensor=accuracy)

        # merge summary op
        summary_op = tf.summary.merge_all()

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
            for i in range(FLAGS.epoch):
                for j in range(FLAGS.step_per_epoch):

                    train_batch_x, train_batch_y = train_image_iter.next()
                    _, train_summary_str, train_loss, train_accuracy = sess.run([train_op, summary_op, total_loss,
                                                                                 accuracy], feed_dict={
                                                                            self.input_image: train_batch_x,
                                                                            self.labels: train_batch_y,
                                                                            self.lr: FLAGS.init_lr})

                    train_step = i*FLAGS.step_per_epoch+j

                    train_writer.add_summary(summary=train_summary_str, global_step=train_step)

                    if train_step % FLAGS.report_step == 0:
                        valid_summary_loss_scalar, valid_summary_accuracy_scalar, valid_loss, valid_accuracy = \
                            sess.run([loss_scalar, accuracy_scalar, total_loss, accuracy],
                                     feed_dict={
                                                self.input_image: validate_data,
                                                self.labels: validate_label,
                                                self.lr: FLAGS.init_lr})
                        test_writer.add_summary(summary=valid_summary_loss_scalar, global_step=train_step)
                        test_writer.add_summary(summary=valid_summary_accuracy_scalar, global_step=train_step)
                        print('-----------------------')
                        print('validation:  step: %d, loss: %f, accuracy: %f'
                              % (train_step, valid_loss, valid_accuracy))
                        print('-----------------------')

                    print('step: %d, loss: %f, accuracy: %f' % (train_step, train_loss, train_accuracy))

                    if train_step == FLAGS.first_lr_decay_step or train_step == FLAGS.second_lr_decay_step:
                        FLAGS.init_lr *= FLAGS.lr_decay
                        print('learn rate has changed now! lr: %f' % FLAGS.init_lr)
