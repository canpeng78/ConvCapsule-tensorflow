"""
Copyright (C) 2017 Can Peng
License: MIT Software License
Author: Can Peng (can.peng78@gmail.com)
"""

from __future__ import division
import tensorflow as tf
import numpy as np
from six.moves import xrange
import math
import types

try:
    im_summary = tf.image_summary
    scl_summary = tf.scalar_summary
    hist_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    merge_all = tf.merge_all_summaries
    SummaryWriter = tf.train.SummaryWriter
except:
    im_summary = tf.summary.image
    scl_summary = tf.summary.scalar
    hist_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    merge_all = tf.summary.merge_all
    SummaryWriter = tf.summary.FileWriter

def conv2d_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, scope=None, weight_name='w', bias_name='b', stddev=0.005, reuse=False, return_vars=False):
    if isinstance(stride, types.IntType):
        stride=[stride]
        
    while len(stride)<2:    
        stride.append(stride[-1]) 
    with tf.variable_scope(scope) as scope:
        b = tf.get_variable(
            bias_name,
            shape=filter_shape[-1],
            initializer=tf.constant_initializer(0.))
        if reuse == True:
            scope.reuse_variables()
        w = tf.get_variable(
            weight_name,
            shape=filter_shape,
            initializer=tf.random_normal_initializer(0., stddev))
      
        conv = tf.nn.conv2d( bottom, w, [1,stride[0],stride[1],1], padding=padding)
        o = activation(tf.nn.bias_add(conv, b))
        if not return_vars:
            return o
        else:
            return o, w, b

def deconv2d_layer(bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, scope=None, weight_name='w', bias_name='b', stddev=0.005, reuse=False, return_vars=False):
    if isinstance(stride, types.IntType):                                                                                                                                     
        stride=[stride] 

    while len(stride)<2:       
        stride.append(stride[-1])   
    with tf.variable_scope(scope) as scope:
        b = tf.get_variable(
            bias_name,
            shape=filter_shape[-2],
            initializer=tf.constant_initializer(0.))
        if reuse == True:
            scope.reuse_variables()
        w = tf.get_variable(
            weight_name,
            shape=filter_shape,
            initializer=tf.random_normal_initializer(0., stddev))
        
        deconv = tf.nn.conv2d_transpose( bottom, w, output_shape, [1,stride[0],stride[1],1], padding=padding)
        o = activation(tf.nn.bias_add(deconv, b))
        if not return_vars:
            return o
        else:
            return o, w, b

def conv3d_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, scope=None, weight_name='w', bias_name='b', stddev=0.005, reuse=False, return_vars=False):
    if isinstance(stride, types.IntType):                                                                                                                                     
        stride=[stride] 

    while len(stride)<3:
        stride.append(stride[-1])
    with tf.variable_scope(scope) as scope:
        b = tf.get_variable(
            bias_name,
            shape=filter_shape[-1],
            initializer=tf.constant_initializer(0.))
        if reuse == True:
            scope.reuse_variables()
        w = tf.get_variable(
            weight_name,
            shape=filter_shape,
            initializer=tf.random_normal_initializer(0., stddev))
    
        conv = tf.nn.conv3d(bottom, w, [1,stride[0],stride[1],stride[2],1], padding=padding)
        o = activation(tf.nn.bias_add(conv, b))
        if not return_vars:
            return o
        else:
            return o, w, b

def deconv3d_layer(bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, scope=None, weight_name='w', bias_name='b', stddev=0.005, reuse=False, return_vars=False):
    if isinstance(stride, types.IntType):                                                                                                                                     
        stride=[stride] 
    while len(stride)<3:           
        stride.append(stride[-1])   
    with tf.variable_scope(scope) as scope:
        b = tf.get_variable(
            bias_name,
            shape=filter_shape[-2],
            initializer=tf.constant_initializer(0.))
        if reuse == True:
            scope.reuse_variables()
        w = tf.get_variable(
            weight_name,
            shape=filter_shape,
            initializer=tf.random_normal_initializer(0., stddev))
    
        deconv = tf.nn.conv3d_transpose( bottom, w, output_shape, [1,stride[0],stride[1],stride[2],1], padding=padding)
        o = activation(tf.nn.bias_add(deconv, b))
        if not return_vars:
            return o
        else:
            return o, w, b

def leaky_relu(bottom, leak=0.1, name="lrelu"):
    return tf.maximum(leak*bottom, bottom)

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True, disable=False):
        if disable == True:
            with tf.variable_scope(self.name):
                return tf.identity(x,name='copy')
        else:
            return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum, 
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=train,
                                                scope=self.name)

def linear(input_, output_size, scope=None, weight_name='w', bias_name='b', stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(weight_name, [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(bias_name, [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
