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

def googlenet(features, labels, mode):
    inputs = tf.reshape(features['x'], [-1,224,224,3])
    inputs = conv(inputs, 64, 7, 2)
    inputs = pool(inputs, 3, "max", 2, "SAME")
    inputs = conv(inputs, 192, 3, 1)
    inputs = pool(inputs, 3, "max", 2, "SAME")
    inputs = inception(inputs, [64, (96,128), (16, 32), 32])
    inputs = inception(inputs, [128, (128,192), (32,96), 64])
    inputs = pool(inputs, 3, "max", 2, "SAME")
    inputs = inception(inputs, [192, (96, 208), (16, 48), 64])
    inputs, pred1 = inception(inputs, [160, (112, 224), (24, 64), 64], pred_layer=True)
    inputs = inception(inputs, [128, (128, 256), (24, 64), 64])
    inputs = inception(inputs, [112, (144, 288), (32, 64), 64])
    inputs, pred2 = inception(inputs, [256, (160, 320), (32, 128), 128], pred_layer=True)
    inputs = pool(inputs, 3, "max", 2, "SAME")
    inputs = inception(inputs, [256, (160, 320), (32, 128), 128])
    inputs = inception(inputs, [384, (192, 384), (48, 128), 128])
    inputs = pool(inputs, 7, "average", 1, "VALID")
    inputs = tf.contrib.layers.flatten(inputs)
    inputs = tf.layers.dense(inputs=inputs, units=1024)
    inputs = tf.layers.dropout(inputs=inputs,rate=0.4,training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs = inputs, units = 62)
    predictions = {
        "classes": tf.argmax(input=inputs, axis=1),
        "probabilities": tf.nn.softmax(inputs,
                                       name="softmax_tensor")}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predicitions=predictions)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
    loss1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                            logits=pred1)
    loss2 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                            logits=pred2)
    loss3 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                            logits=logits)
    weighted_loss = .3*(loss1 + loss2) + loss3
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=weighted_loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=weighted_loss,
                                          train_op=train_op)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                      loss=loss3,
                                      eval_metric_ops=eval_metric_ops)
