# Code largely from TensorFlow mnist tutorial https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# Feedforward params
NUM_CLASSES = 2

# Input size
INPUT_SIZE = 15117

# TODO: more easily paramatrizable layers
def inference(inputs, hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units):
    # Builds the graph as far as needed to return the tensor that would contain the output predicitions.

    # Hidden layer 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([INPUT_SIZE, hidden1_units], 
            stddev=1.0/math.sqrt(float(INPUT_SIZE))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

    # Hidden layer 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
            stddev=1.0/math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
            name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    
    # Hidden layer 3
    with tf.name_scope('hidden3'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units, hidden3_units],
            stddev=1.0/math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden3_units]),
            name='biases')
        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases) 

    # Hidden layer 4
    with tf.name_scope('hidden4'):
        weights = tf.Variable(tf.truncated_normal([hidden3_units, hidden4_units],
            stddev=1.0/math.sqrt(float(hidden3_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden4_units]),
            name='biases')
        hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases) 

    # Hidden layer 5
    with tf.name_scope('hidden5'):
        weights = tf.Variable(tf.truncated_normal([hidden4_units, hidden5_units],
            stddev=1.0/math.sqrt(float(hidden4_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden5_units]),
            name='biases')
        hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases) 
        
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden5_units, NUM_CLASSES],
            stddev=1.0/math.sqrt(float(hidden5_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
            name='biases')
        logits = tf.matmul(hidden5, weights) + biases

    return logits

def loss(logits, labels):
    # Calculates the loss from the logits and the labels

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
    # Sets up training ops
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    # Evaluate the quality of the logits at predicting the label
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))