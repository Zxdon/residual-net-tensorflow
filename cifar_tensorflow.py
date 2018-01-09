# coding=UTF-8
# coder: Zxdon
# github: https://github.com/Zxdon/residual-net-tensorflow-self

# -----------------------------------------------------------------

from model_train import *


def main(_):
    train = Train()
    train.train()


if __name__ == '__main__':
    tf.app.run()

