"""
Copyright (C) 2017 Can Peng
License: MIT Software License
Author: Can Peng (can.peng78@gmail.com)
"""

import types
from ops import *
from utils import *


class CapsuleConv2d(object):
    """
    Convolutional Capsule Layer
    Return a 5D tensor [batch_size, height, width, capsule_len, channels]
    """

    def __init__(self, capsize, routing=None, rt_iternum=3, name=None):
        self.capsize = capsize
        self.routing = routing
        self.rt_iternum = rt_iternum
        self.name = name

    def __call__(self, input, outchannels, kernelsize=None, stride=None, padding="VALID", reuse=False, weight_name='w',
                 stddev=0.005):
        """
        input is a 4D tensor [batch_size, height, width, channels] if routing is None for primary capsule layer
        input is a 5D tensor [batch_size, height, width, capsule_len, channels] if routing is not None
        outchannels is an integer
        kernelsize is a list [filter_height, filter_width, in_channels]
        """

        if isinstance(stride, types.IntType):
            stride = [stride]
        while len(stride) < 2:
            stride.append(stride[-1])

        self.out_channels = outchannels
        self.kernel_size = kernelsize
        self.stride = stride
        self.padding = padding

        inshape = input.get_shape().as_list()

        if self.routing is None:
            tot_channels = self.out_channels * self.capsize
            filter_shape = kernelsize
            filter_shape.append(tot_channels)
            with tf.variable_scope(self.name) as scope:
                b = tf.get_variable(
                "bias",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))
                if reuse:
                    scope.reuse_variables()
                w = tf.get_variable(
                    weight_name,
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., stddev))
                conv = tf.nn.bias_add(tf.nn.conv2d(input, w, [1, stride[0], stride[1], 1], padding=padding), b)
                convshape = conv.get_shape().as_list()
                capsules = tf.reshape(conv, shape=(
                    convshape[0], convshape[1], convshape[2], self.capsize, self.out_channels))
                capsules = self.squash(capsules)
                return capsules
        elif self.routing == "Dynamic":
            # Do affine transform for each input channel
            capsule_array = []
            for i in range(0, inshape[-1]):
                filter_shape = [1, 1, inshape[-2], self.capsize*self.out_channels]
                with tf.variable_scope(self.name + '_chan_' + str(i)) as scope:
                    if reuse:
                        scope.reuse_variables()
                    w = tf.get_variable(
                        weight_name,
                        shape=filter_shape,
                        initializer=tf.random_normal_initializer(0., stddev))
    
                    chan = tf.nn.conv2d(input[:, :, :, :, i], w, [1, 1, 1, 1], padding="VALID")
                    # chan is in shape of [batch_size, height, width, self.capsize*self.out_channels]
                    capsule_array.append(tf.reshape(chan, shape=[inshape[0], inshape[1], inshape[2],
                                                                 self.capsize, self.out_channels]))
            # Do routing convolutionally
            patches, gridsz = patches2d(inshape[1:], self.kernel_size, self.stride, self.padding)
            out_caplayer = []
            for patch in patches:
                capsules = []
                for cap in capsule_array:
                    capptch = cap[:, patch[0]:patch[2], patch[1]:patch[3], :, :]
                    capshape = capptch.get_shape().as_list()
                    capptch = tf.reshape(capptch, shape=(capshape[0], -1, capshape[-2], capshape[-1]))
                    capsules.append(capptch)
                try:
                    capsules = tf.concat(capsules, 1)
                except:
                    capsules = tf.concat(1, capsules)# shape [batch_size, num_capsules, self.capsize, self.out_channels]
                out_capsule = tf.expand_dims(self.dynamic_routing(capsules), 1)  # shape [batch_size, 1, self.capsize, self.out_channels]
                out_caplayer.append(out_capsule)
            try:
                out_caplayer = tf.concat(out_caplayer, 1)
            except:
                out_caplayer = tf.concat(1, out_caplayer)
            out_caplayer = tf.reshape(out_caplayer, shape=(
                tf.shape(out_caplayer)[0], gridsz[0], gridsz[1], self.capsize, self.out_channels))
            return out_caplayer

    def dynamic_routing(self, incaps, random_init=False):
        """
        Do dynamic routing
        """
        # incaps in shape of [batch_size, num_capsules, self.capsize, self.out_channels]
        inshape = incaps.get_shape().as_list()
        incaps_grdstop = tf.stop_gradient(incaps)
        if not random_init:
            b = tf.constant(0.0, shape=inshape)
        else:
            db = tf.random_uniform([inshape[0], inshape[1], 1, inshape[3]], minval=-0.5, maxval=0.5)
            b = tf.tile(db, [1, 1, self.capsize, 1])
        for it in xrange(self.rt_iternum):
            self.c = tf.nn.softmax(b)
            if it == self.rt_iternum - 1:
                s = tf.multiply(incaps, self.c)
                s = tf.reduce_sum(s, axis=1)  # s is in shape of [batch_size, self.capsize, self.out_channels]
                v = self.squash(s)
            else:
                s = tf.multiply(incaps_grdstop, self.c)
                s = tf.reduce_sum(s, axis=1)
                v = self.squash(s)  # v in shape [batch_size, self.capsize, self.out_channels]
                v_tiled = tf.tile(tf.expand_dims(v, 1), [1, inshape[1], 1, 1])
                db = tf.reduce_sum(tf.multiply(incaps_grdstop, v_tiled), 2, keep_dims=True)
                b += tf.tile(db, [1, 1, self.capsize, 1])
        return v

    @staticmethod
    def squash(input, axis=-2):
        sqrsum = tf.reduce_sum(tf.square(input), axis, keep_dims=True)
        vnorm = tf.sqrt(sqrsum)
        scale = vnorm / (1 + sqrsum)
        output = scale * input
        return output
