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

from wavenet import WaveNetModel, AudioReader, optimizer_factory, mu_law_decode, mu_law_encode

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

# Misc Params
VIEW_INITIAL_WHITE = False

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
    parser.add_argument('--view_initial_white', type=bool, default=VIEW_INITIAL_WHITE,
                        help='View plot of initial input white news. Default: '
                             + str(VIEW_INITIAL_WHITE) + '.')
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

def main():

    with tf.Graph().as_default():
        coord = tf.train.Coordinator()
        sess = tf.Session()

        batch_size = 10
        hidden1_units = 5202
        hidden2_units = 2601
        hidden3_units = 1300
        hidden4_units = 650
        hidden5_units = 325
        max_training_steps = 5

        global_step = tf.Variable(0, name='global_step', trainable=False)
        initial_training_learning_rate = 3e-2
        training_learning_rate = tf.train.exponential_decay(initial_training_learning_rate, global_step, 100, 0.9, staircase=True)

        inputs_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        logits = ffnn.inference(inputs_placeholder, hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units)
        loss = ffnn.loss(logits, labels_placeholder)
        train_op = ffnn.training(loss, training_learning_rate, global_step)
        eval_correct = ffnn.evaluation(logits, labels_placeholder)

        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        summary_writer = tf.summary.FileWriter('./logdir', sess.graph)

        sess.run(init)

        args = get_arguments()

        if args.restore_from != None:
            restore_from = args.restore_from
            print("Restoring from: ")
            print(restore_from)

        else:
            restore_from = ""

        try:
            saved_global_step = load(saver, sess, restore_from)
            if saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1
            else:
                counter = saved_global_step % label_batch_size

        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        # TODO: Find a more robust way to find different data sets

        # Training data 
        directory = './sampleTrue'
        reader = AudioReader(directory, coord, sample_rate = 16000, gc_enabled=False, receptive_field=5117, sample_size=10000, silence_threshold=0.05)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)

        directory = './sampleFalse'
        reader2 = AudioReader(directory, coord, sample_rate = 16000, gc_enabled=False, receptive_field=5117, sample_size=10000, silence_threshold=0.05)
        threads2 = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader2.start_threads(sess)

        total_loss = 0
        for step in range(saved_global_step + 1, max_training_steps):
            start_time = time.time()

            batch_data = []
            label_data = []

            if (step % 100 == 0):
                print('Current learning rate: %6f' % (sess.run(training_learning_rate)))

            for b in range(batch_size):
                label = randint(0, 1)

                if label == 1:
                    data = sess.run(reader.dequeue(1))

                    while (len(data[0]) < ffnn.INPUT_SIZE):
                        data = sess.run(reader.dequeue(1))
                else:
                    data = sess.run(reader2.dequeue(1))

                    while (len(data[0]) < ffnn.INPUT_SIZE):
                        data = sess.run(reader2.dequeue(1))

                data = np.array(data[0])
                mean = np.mean(data)
                std = np.std(data)

                standardized = []

                for d in data:
                    standardized.append(float(d-mean)/std)

                batch_data.append(standardized)
                label_data.append(label)

            feed_dict = fill_feed_dict(batch_data, label_data, inputs_placeholder, labels_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time
            total_loss = total_loss + loss_value

            print('Step %d: loss = %.7f (%.3f sec)' % (step, loss_value, duration))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
            '''
            if step % 100 == 0 or (step + 1) == max_training_steps:
                average = total_loss / (step + 1)
                print('Cumulative average loss: %6f' % (average))
                # TODO: Update train script to add data to new directory
                checkpoint_file = os.path.join('./logdir/init-train/', 'model.ckpt')
                print("Generating checkpoint file...")
                saver.save(sess, checkpoint_file, global_step=step)
            '''

        testData = sess.run(reader.dequeue(1))
        testData = testData[0]
        testData = mu_law_encode(testData, 256)

        print(testData)

        # Lambda for white noise sampler
        gi_sampler = get_generator_input_sampler()

        # White noise generation and verification

        # White noise generator params
        white_mean = 0
        white_sigma = 1
        white_length = 20234

        white_noise = gi_sampler(white_mean, white_sigma, white_length)
        if args.view_initial_white:
            plt.plot(white_noise)
            plt.ylabel('Amplitude')
            plt.xlabel('Time')
            plt.show()

        # Load parameters from wavenet params json file
        with open(args.wavenet_params, 'r') as f:
            wavenet_params = json.load(f)  

        # Initialize generator WaveNet
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

        # Calculate loss for white noise input
        result = G.loss(input_batch=tf.convert_to_tensor(white_noise, dtype=np.float32), name='generator')
        
        init = tf.global_variables_initializer()
        sess.run(init)

        gi_sampler = get_generator_input_sampler()

        # White noise generator params
        white_mean = 0
        white_sigma = 1
        white_length = 20234

        white_noise = gi_sampler(white_mean, white_sigma, white_length)

        loss = G.loss(input_batch=tf.convert_to_tensor(white_noise, dtype=np.float32), name='generator')

        samples = tf.placeholder(tf.int32)

        if args.fast_generation:
            next_sample = G.predict_proba_incremental(samples, args.gc_id)
        else:
            next_sample = G.predict_proba(samples, args.gc_id)

        if args.fast_generation:
            sess.run(tf.global_variables_initializer())
            sess.run(G.init_ops)

        decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

        quantization_channels = wavenet_params['quantization_channels']
        
        '''
        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))
        '''
        waveform = [0]

        last_sample_timestamp = datetime.now()
        for step in range(15117):
            if args.fast_generation:
                outputs = [next_sample]
                outputs.extend(G.push_ops)
                window = waveform[-1]
            else:
                if len(waveform) > G.receptive_field:
                    window = waveform[-G.receptive_field:]
                else:
                    window = waveform
                outputs = [next_sample]

            # Run the WaveNet to predict the next sample.
            prediction = sess.run(outputs, feed_dict={samples: window})[0]

            # Scale prediction distribution using temperature.
            np.seterr(divide='ignore')
            scaled_prediction = np.log(prediction) / 1
            scaled_prediction = (scaled_prediction -
                                 np.logaddexp.reduce(scaled_prediction))
            scaled_prediction = np.exp(scaled_prediction)
            np.seterr(divide='warn')

            sample = np.random.choice(
                np.arange(quantization_channels), p=scaled_prediction)
            waveform.append(sample)

        del waveform[0]
        print(waveform)

        '''
        X = tf.placeholder(tf.float32, shape=[None, ffnn.INPUT_SIZE], name='X')

        D_logit_real = ffnn.inference(X, hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units)
        D_real = tf.nn.sigmoid(D_logit_real)

        D_logit_fake = ffnn.inference(G_sample[0], hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units)
        D_fake = tf.nn.sigmoid(D_logit_fake)

        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

        D_variables = tf.trainable_variables(scope='discriminator')
        G_variables = tf.trainable_variables(scope='wavenet')

        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_variables)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_variables)

        for step in NUM_EPOCHS:
            X_mb = []

            data = sess.run(reader.dequeue(1))

            while (len(data[0]) < ffnn.INPUT_SIZE):
                data = sess.run(reader.dequeue(1))

            data = np.array(data[0])
            mean = np.mean(data)
            std = np.std(data)

            standardized = []

            for d in data:
                standardized.append(float(d-mean)/std)

            X_mb.append(standardized)

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb})
            _, G_loss_curr = sess.run([G_solver, G_loss])
        '''        

if __name__ == '__main__':
    main()