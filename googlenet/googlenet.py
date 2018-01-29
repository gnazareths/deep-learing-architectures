from __future__ import absolute_import, division, print_function
from utilities import *

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
