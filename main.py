"""
Copyright (C) 2017 Can Peng
License: MIT Software License
Author: Can Peng (can.peng78@gmail.com)
"""

import os

from model import Caps3Layers
from utils import *

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Start learning rate of for adam [0.0002]")
flags.DEFINE_integer("decay_steps", 1000, "Decay steps for learning rate decaying")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate for learning rate [0.96]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_float("weight_decay", 0.00001, "Weight decaying rate [0.0001]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 28, "The height of image [128]")
flags.DEFINE_integer("input_width", 28, "The width of image [128]")
flags.DEFINE_string("model_name", "mnistcapsnet_mloss_relu_recon3", "The name of model dir")
flags.DEFINE_string("checkpoint_dir", "/s0/cpeng/models/capsnet/models/",
                    "Directory to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "./logs/", "Directory to save logs")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

data_dir = "./mnist_data/"

def main(_):

    if not os.path.isdir(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        capsnet = Caps3Layers(sess,
                              input_height=FLAGS.input_height,
                              input_width=FLAGS.input_width,
                              batch_size=FLAGS.batch_size,
                              weight_decay=FLAGS.weight_decay,
                              model_name=FLAGS.model_name)

        display_variables()

        if FLAGS.train:
            capsnet.train(FLAGS, data_dir)
        else:
            capsnet.test(FLAGS, data_dir, "images/reconimg3.png")


if __name__ == '__main__':
    tf.app.run()
