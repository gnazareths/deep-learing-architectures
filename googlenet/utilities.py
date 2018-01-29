import os
import cv2
import numpy as np
import skimage.data as sk
import tensorflow as tf

from skimage import transform
from __future__ import absolute_import, division, print_function
from utilities import *

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
mode = tf.estimator.ModeKeys.TRAIN

def batch_norm_relu(inputs, is_training):
    inputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                 #axis=1,
                                 #momentum=_BATCH_NORM_DECAY,
                                 #epsilon=_BATCH_NORM_EPSILON,
                                 center=True,
                                 scale=True,
                                 is_training=is_training,
                                 fused=True,
                                 data_format="NHWC")
    inputs = tf.nn.relu(inputs)
    return inputs

def conv(inputs, n_filters, kernel_size, strides, name=None, padding='SAME'):
    inputs = tf.layers.conv2d(inputs,
                              filters=n_filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.variance_scaling_initializer())
    inputs = batch_norm_relu(inputs, True)
    return inputs

def pool(inputs, size, pooling_type, strides, padding):
    if pooling_type == 'average':
        fn = tf.layers.average_pooling2d
    elif pooling_type == 'max':
        fn = tf.layers.max_pooling2d
    else:
        raise ValueError("pooling_type must be in ['average', 'max']")
    return fn(inputs=inputs,
              pool_size=size,
              strides=strides,
              padding=padding)

def dense(inputs, units):
    inputs = tf.layers.dense(inputs=inputs, units=units)
    return inputs

def dropout(inputs, rate):
    inputs = tf.layers.dropout(inputs=inputs,rate=rate,training=mode == tf.estimator.ModeKeys.TRAIN)
    return inputs

def inception_prediction(inputs):
    inputs = pool(inputs, 5, "average", 3, "VALID")
    inputs = conv(inputs, 128, 1, 1, name="Loss_layer_conv_1x1")
    inputs = dense(inputs, 1024)
    inputs = dropout(inputs, .7)
    inputs = tf.contrib.layers.flatten(inputs)
    logits = dense(inputs, 62)
    #probabilities = tf.nn.softmax(logits,name="softmax_tensor")
    return logits

def inception(inputs, filter_sizes, pred_layer=False):
    f1, f2, f3, f4 = filter_sizes
    i1 = conv(inputs, f1, 1, 1, name="Inception_conv_1x1")
    i2 = conv(inputs, f2[0], 1, 1, name="Inception_conv_3x3_red")
    i3 = conv(inputs, f3[0], 1, 1, name="Inception_conv_5x5_red")
    i2 = conv(i2, f2[1], 3, 1, name="Inception_conv_3x3")
    i3 = conv(i3, f3[1], 5, 1, name="Inception_conv_5x5")
    i4 = pool(inputs, 3, "max", 1, "SAME")
    i4 = conv(i4, f4, 1, 1, name="Inception_conv_from_pool")
    concat = tf.concat([i1,i2,i3,i4], 3)
    if pred_layer:
        pred = inception_prediction(inputs)
        return concat, pred
    return concat
