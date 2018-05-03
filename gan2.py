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

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import ffnn

from wavenet import WaveNetModel, AudioReader, optimizer_factory, mu_law_decode

# GAN params
NUM_EPOCHS = 10

# Wavenet params
BATCH_SIZE = 1
DATA_DIRECTORY = './VCTK-Corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 3e-2
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False

def placeholder_inputs(batch_size):
    # Generate placeholder variables for input tensors
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, ffnn.INPUT_SIZE), name='inputs_placeholder')
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='labels_placeholder')
    return inputs_placeholder, labels_placeholder

def convert_dequeue_to_list(dequeued):
    data = []
    dequeued = dequeued[0]

    for data_point in dequeued:
        data.append(data_point[0])

    return data

def dequeue_to_list_standardize(dequeued):
    data = np.array(dequeued)
    mean = np.mean(data)
    std = np.std(data)

    standardized = []

    for d in data:
        standardized.append(float(d-mean)/std)

    return standardized

def fill_feed_dict(batch_data, label_data, inputs_pl, labels_pl):
    # Feed dict for placeholders from placeholder_inputs()

    feed_dict = {
        inputs_pl: batch_data,
        labels_pl: label_data
    }

    return feed_dict

def get_generator_input_sampler():
    return lambda mu, sigma, n: np.random.normal(mu, sigma, size=n)

def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

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

def manual_mu_law_encode(signal, quantization_channels):
    # Manual mu-law companding and mu-bits quantization
    mu = quantization_channels - 1

    magnitude = np.log1p(mu * np.abs(signal)) / np.log1p(mu)
    signal = np.sign(signal) * magnitude

    # Map signal from [-1, +1] to [0, mu-1]
    signal = (signal + 1) / 2 * mu + 0.5
    quantized_signal = signal.astype(np.int32)

    return quantized_signal

def process(samples, quantization_channels, encode):
    if encode:
        encoded = manual_mu_law_encode(samples, quantization_channels)
    else:
        encoded = samples
    '''
    standardized = standardize(encoded)
    normalized = normalize(standardized)

    return normalized
    '''

    # normalized = normalize(encoded)
    # standardized = standardize(normalized)

    standardized = standardize(encoded)

    return standardized

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from_init', type=str, default=None,
                        help='Directory in which to restore the initialization from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    parser.add_argument(
        '--fast_generation',
        type=_str_to_bool,
        default=True,
        help='Use fast generation')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    return parser.parse_args()

def discriminator(x):
    D_h1 = tf.nn.leaky_relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_h4 = tf.nn.leaky_relu(tf.matmul(D_h3, D_W4) + D_b4)
    D_h5 = tf.nn.leaky_relu(tf.matmul(D_h4, D_W5) + D_b5)

    D_logit = tf.matmul(D_h5, D_W6) + D_b6
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

with tf.Graph().as_default():
    coord = tf.train.Coordinator()
    sess = tf.Session()

    '''
    w1 = 11000
    w2 = 8000
    w3 = 4000
    w4 = 2000
    w5 = 1000
    w6 = 500
    w7 = 1
    '''
    w1 = 22000
    w2 = 18335
    w3 = 14668
    w4 = 11001
    w5 = 7334
    w6 = 3667
    w7 = 1

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

    args = get_arguments()

    # Load parameters from wavenet params json file
    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)  

    quantization_channels = wavenet_params['quantization_channels']

    # Intialize generator WaveNet
    G = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"],
        initial_filter_width=wavenet_params["initial_filter_width"])

    gi_sampler = get_generator_input_sampler()

    # White noise generator params
    white_mean = 0
    white_sigma = 1
    white_length = ffnn.INPUT_SIZE

    white_noise = gi_sampler(white_mean, white_sigma, white_length)
    white_noise = process(white_noise, quantization_channels, 1)
    white_noise_t = tf.convert_to_tensor(white_noise)

    # initialize generator
    w_loss, w_prediction = G.loss(input_batch=white_noise_t, name='generator')

    theta_G = tf.trainable_variables(scope='wavenet')

    X = tf.placeholder(tf.float32, shape=[None, w1], name='X')

    init = tf.global_variables_initializer()
    sess.run(init)

    D_real, D_logit_real = discriminator(X)
    D_fake, D_logit_fake = discriminator(G_sample)

    levels = []
    for i in range(quantization_channels):
        levels.append(i)

    print(first[0] * levels)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake), logits=D_logit_fake))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))

    # Only update D(X)'s parameters, so var_list = theta_D
    D_solver = tf.train.GradientDescentOptimizer(learning_rate=3e-3).minimize(D_loss, var_list=theta_D)
    # Only update G(X)'s parameters, so var_list = theta_G
    G_solver = tf.train.GradientDescentOptimizer(learning_rate=3e-3).minimize(G_loss, var_list=theta_G)