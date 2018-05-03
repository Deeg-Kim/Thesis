from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from random import *
import argparse
import json
import os
import sys
import time

import librosa
import tensorflow as tf

import numpy as np

from wavenet import AudioReader, mu_law_decode

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def standardize(samples):
    mean = np.mean(samples)
    std = np.std(samples)
    
    standardized = []

    for d in samples:
        standardized.append(float(d-mean)/std)

    return standardized

def normalize(samples):
    maximum = np.max(samples)
    minimum = np.min(samples)

    normalized = []

    for d in samples:
        normalized.append((d-minimum) * float(1/(maximum - minimum)))

    return normalized

def process(samples):
    # standardized = standardize(samples)

    quick = []
    for d in samples:
        quick.append(d[0])

    standardized = quick

    return standardized

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

w1 = 22000
w2 = 18335
w3 = 14668
w4 = 11001
w5 = 7334
w6 = 3667
w7 = 1

'''
w1 = 11000
w2 = 8000
w3 = 4000
w4 = 2000
w5 = 1000
w6 = 500
w7 = 1
'''

# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, w1], name='X')

D_W1 = tf.Variable(xavier_init([w1, w2]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[w2]), name='D_b1')

D_W2 = tf.Variable(xavier_init([w2, w3]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[w3]), name='D_b2')

D_W3 = tf.Variable(xavier_init([w3, w4]), name='D_W3')
D_b3 = tf.Variable(tf.zeros(shape=[w4]), name='D_b3')

D_W4 = tf.Variable(xavier_init([w4, w5]), name='D_W4')
D_b4 = tf.Variable(tf.zeros(shape=[w5]), name='D_b4')

D_W5 = tf.Variable(xavier_init([w5, w6]), name='D_W5')
D_b5 = tf.Variable(tf.zeros(shape=[w6]), name='D_b5')

D_W6 = tf.Variable(xavier_init([w6, w7]), name='D_W6')
D_b6 = tf.Variable(tf.zeros(shape=[w7]), name='D_b6')

theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_W6, D_b1, D_b2, D_b3, D_b4, D_b5, D_b6]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, w6]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[w6]), name='G_b1')

G_W2 = tf.Variable(xavier_init([w6, w5]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[w5]), name='G_b2')

G_W3 = tf.Variable(xavier_init([w5, w4]), name='G_W3')
G_b3 = tf.Variable(tf.zeros(shape=[w4]), name='G_b3')

G_W4 = tf.Variable(xavier_init([w4, w3]), name='G_W4')
G_b4 = tf.Variable(tf.zeros(shape=[w3]), name='G_b4')

G_W5 = tf.Variable(xavier_init([w3, w2]), name='G_W5')
G_b5 = tf.Variable(tf.zeros(shape=[w2]), name='G_b5')

G_W6 = tf.Variable(xavier_init([w2, w1]), name='G_W6')
G_b6 = tf.Variable(tf.zeros(shape=[w1]), name='G_b6')

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_W6, G_b1, G_b2, G_b3, G_b4, G_b5, G_b6]

def generator(z):
    G_h1 = tf.nn.leaky_relu(tf.matmul(z, G_W1) + G_b1)
    G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.leaky_relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_h4 = tf.nn.leaky_relu(tf.matmul(G_h3, G_W4) + G_b4)
    G_h5 = tf.nn.leaky_relu(tf.matmul(G_h4, G_W5) + G_b5)
    
    G_log_prob = tf.matmul(G_h5, G_W6) + G_b6
    G_prob = tf.nn.sigmoid(G_log_prob)    

    return G_prob

def discriminator(x):
    D_h1 = tf.nn.leaky_relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_h4 = tf.nn.leaky_relu(tf.matmul(D_h3, D_W4) + D_b4)
    D_h5 = tf.nn.leaky_relu(tf.matmul(D_h4, D_W5) + D_b5)

    D_logit = tf.matmul(D_h5, D_W6) + D_b6
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake), logits=D_logit_fake))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.GradientDescentOptimizer(learning_rate=3e-3).minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.GradientDescentOptimizer(learning_rate=3e-3).minimize(G_loss, var_list=theta_G)

coord = tf.train.Coordinator()
sess = tf.Session()

directory = './sampleTrue'
reader = AudioReader(directory, coord, sample_rate = 22000, gc_enabled=False, receptive_field=1000, sample_size=21000, silence_threshold=0.05)
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
reader.start_threads(sess)

init = tf.global_variables_initializer()
sess.run(init)

prevA = []
for it in range(1000):
    batch_data = []

    start_time = time.time()

    data = sess.run(reader.dequeue(1))
    while (len(data[0]) < w1):
        data = sess.run(reader.dequeue(1))

    data = np.array(data[0])
    samples = process(data)
    batch_data.append(samples)

    for g_batch in range(10):
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(1, 100)})

    for d_batch in range(1):
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_data, Z: sample_Z(1, 100)})

    '''
    waveform = np.reshape(sess.run(G_sample, feed_dict={Z: sample_Z(1, 100)}), [w1])
    print(waveform)
    nextA = []
    for i in range(22000):
        nextA.append(waveform[i])

    print("real logit")
    print(sess.run(D_real, feed_dict={X: batch_data, Z: sample_Z(1, 100)}))

    print("fake logit")
    print(sess.run(D_fake, feed_dict={Z: sample_Z(1, 100)}))

    print("Equal?")
    print(np.array_equal(prevA, nextA))
    prevA = nextA
    '''
    
    duration = time.time() - start_time

    if (it % 20 == 0):
        waveform = []
        waveform = np.reshape(sess.run(G_sample, feed_dict={Z: sample_Z(1, 100)}), [w1])
        print(waveform)
        name = '5-3simplegenerate-' + str(it) + '.wav'
        write_wav(waveform, 22000, name)

    print('Step %d: 1st D loss = %.7f, 10th G loss = %.7f (%.3f sec)' % (it, D_loss_curr, G_loss_curr, duration))

samples = tf.placeholder(tf.int32)
decode = mu_law_decode(samples, 256)

waveform = []
waveform = np.reshape(sess.run(G_sample, feed_dict={Z: sample_Z(1, 100)}), [w1])

write_wav(waveform, 22000, '5-3simplegenerate.wav')