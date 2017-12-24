"""
Copyright (C) 2017 Can Peng
License: MIT Software License
Author: Can Peng (can.peng78@gmail.com)
"""

from __future__ import division
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from math import *
    
def patches2d(input_size, kernel_size, strides, padding="VALID"):
    '''
    Create 2D patches based on the input size and the kernel size for convolutional routing
    Return a list of the starting corner and ending corner of patches, and
    grid size of the patch array
    '''
    height = input_size[0]
    width = input_size[1]
    kh = kernel_size[0]
    kw = kernel_size[1]
    sh = strides[0]
    sw = strides[1]
    if padding == "VALID":
        patches = []
        gridsz = []
        col = 0
        i = 0
        while i+kh <= height:
            row = 0
            j = 0
            while j+kw <= width:
                patches.append([i, j, i+kh, j+kw])
                j += sw
                row += 1
            if i == 0:
                gridsz.append(row)
            i += sh
            col += 1
        gridsz.insert(0, col)
        return patches, gridsz
    elif padding == "SAME":
        ns_h = ceil((height - kh) / sh)
        ns_w = ceil((width - kw) / sw)
        padded_h = ns_h * sh + kh
        padded_w = ns_w * sw + kw
        strtpd_h = (padded_h - height) // 2
        strtpd_w = (padded_w - width) // 2
        patches = []
        gridsz = []
        col = 0
        i = 0
        while i+kh < padded_h:
            row = 0
            j = 0
            while j+kw < padded_w:
                patches.append([int(max(i-strtpd_h, 0)), int(max(j-strtpd_w, 0)),
                                int(min(i-strtpd_h+kh, height)), int(min(j-strtpd_w+kw, width))])
                j += sw
                row += 1
            if i == 0:
                gridsz.append(row)
            i += sh
            col += 1
        gridsz.insert(0, col)
        return patches, gridsz
    
def load_mnist(data_dir, vec_label=True, is_train=True):
    if is_train:
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)
        trY = np.asarray(trY)
        if vec_label:
            y_vec = np.zeros([len(trY), 10], dtype=np.float)
            for i, label in enumerate(trY):
                y_vec[i, int(label)] = 1.0
            return trX/255., y_vec
        else:
            return trX/255., trY
    else:
        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)
        teY = np.asarray(teY)
        if vec_label:
            y_vec = np.zeros([len(teY), 10], dtype=np.float)
            for i, label in enumerate(teY):
                y_vec[i, int(label)] = 1.0
            return teX/255., y_vec
        else:
            return teX/255., teY


def display_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    mimg = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for id, img in enumerate(images):
        i = id % size[1]
        j = id // size[1]
        mimg[j*h:j*h+h, i*w:i*w+w] = img
    return np.squeeze(mimg)
