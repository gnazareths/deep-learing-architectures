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

def conv(inputs, n_filters, kernel_size, strides=1, name=None):
    inputs = tf.layers.conv2d(inputs,
                              filters=n_filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=('SAME' if strides == 1 else 'VALID'),
                              activation=tf.nn.relu,
                              kernel_initializer=tf.variance_scaling_initializer())
    return inputs

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

def block(inputs, n_filters, is_training, projection_shortcut, strides):
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs, n_filters)
    inputs = conv(inputs, n_filters, (1,1), strides=1, name="convolution_1")
    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv(inputs, n_filters, (3,3), strides=1, name="convolution_2")
    inputs = batch_norm_relu(inputs, is_training)
    inputs = conv(inputs, (4*n_filters if projection_shortcut else n_filters), (1,1), strides=strides, name="convolution_3")
    inputs = batch_norm_relu(inputs, is_training)
    return inputs + shortcut

def projection_shortcut(inputs, n_filters):
    return conv(inputs=inputs,
                n_filters=4*n_filters,
                kernel_size=1,
                strides=1)

def pool(inputs, size, pooling_type):
    if pooling_type == 'average':
        return tf.layers.average_pooling2d(inputs=inputs,
                                           pool_size=size,
                                           strides=1,
                                           padding='SAME')
    elif pooling_type == 'max':
        return tf.layers.max_pooling2d(inputs=inputs,
                                       pool_size=size,
                                       strides=1,
                                       padding='SAME')
    else:
        raise ValueError("pooling_type must be in ['average', 'max']")

def resnet_01(features, labels, mode):

    inputs = tf.reshape(features['x'], [-1,56,56,3])
    inputs = block(inputs, 16, True, projection_shortcut, 1)
    inputs = pool(inputs, 7, 'max')
    inputs = block(inputs, 32, True, projection_shortcut, 1)
    inputs = pool(inputs, 2, 'max')
    inputs = block(inputs, 32, True, projection_shortcut, 1)
    inputs = pool(inputs, 2, 'max')
    inputs = block(inputs, 32, True, projection_shortcut, 1)
    inputs = pool(inputs, 2, 'max')
    inputs = tf.contrib.layers.flatten(inputs)
    inputs = tf.layers.dense(inputs=inputs, units=1024)
    inputs = tf.layers.dropout(inputs=inputs,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs = inputs, units = 62)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits,
                                       name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predicitions=predictions)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)
