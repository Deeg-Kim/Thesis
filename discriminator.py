from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from random import *
import argparse
import os
import sys
import time

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import ffnn

from wavenet import AudioReader

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
"""
def fill_feed_dict(batch_data, inputs_pl, labels_pl, batch_size, label):
    # Feed dict for placeholders from placeholder_inputs()
    inputs_feed = []
    labels_feed = []

    for batch in range(batch_size):
        list_data = dequeue_to_list_standardize(batch_data)

        inputs_feed.append(list_data)
        labels_feed.append(label)

    inputs_feed = np.array(inputs_feed, dtype=np.float32)

    feed_dict = {
        inputs_pl: inputs_feed,
        labels_pl: labels_feed
    }

    return feed_dict
"""

def fill_feed_dict(batch_data, label_data, inputs_pl, labels_pl):
    # Feed dict for placeholders from placeholder_inputs()

    feed_dict = {
        inputs_pl: batch_data,
        labels_pl: label_data
    }

    return feed_dict

def do_eval(sess, eval_correct, inputs_placeholder, labels_placeholder, batch_data, batch_size, eval_type):
    # Run one epch of evaluation
    true_count = 0

    # TODO: Don't hardcode number of examples
    if eval_type == 'trainingTrue':
        steps_per_epoch = 35
        label = True
    if eval_type == 'trainingFalse':
        steps_per_epoch = 35
        label = False
    elif eval_type == 'validationTrue':
        steps_per_epoch = 9
        label = True
    elif eval_type == 'validationFalse':
        steps_per_epoch = 9
        label = False
    elif eval_type == 'testTrue':
        steps_per_epoch = 11
        label = True
    elif eval_type == 'testFalse':
        steps_per_epoch = 11
        label = False

    num_examples = steps_per_epoch * batch_size

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(batch_data, inputs_placeholder, labels_placeholder, label)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / num_examples

    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

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
    
    parser = argparse.ArgumentParser(description='Simple feedforward neural network')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
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
        max_steps = 1000
        """
        learning_rate = 1e-2
        print('Learning Rate:')
        print(learning_rate)
        print('Layers')
        print(5)
        """
        global_step = tf.Variable(0, name='global_step', trainable=False)
        initial_learning_rate = 4e-2
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 100, 0.95, staircase=True)

        inputs_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        logits = ffnn.inference(inputs_placeholder, hidden1_units, hidden2_units, hidden3_units, hidden4_units, hidden5_units)
        loss = ffnn.loss(logits, labels_placeholder)
        train_op = ffnn.training(loss, learning_rate, global_step)
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
        for step in range(saved_global_step + 1, max_steps):
            start_time = time.time()

            batch_data = []
            label_data = []

            if (step % 100 == 0):
                print('Current learning rate: %6f' % (sess.run(learning_rate)))

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

            if step % 100 == 0 or (step + 1) == max_steps:
                average = total_loss / (step + 1)
                print('Cumulative average loss: %6f' % (average))
                # TODO: Update train script to add data to new directory
                checkpoint_file = os.path.join('./logdir/init-train/', 'model.ckpt')
                print("Generating checkpoint file...")
                saver.save(sess, checkpoint_file, global_step=step)

            """
            if (step + 1) % 500 == 0
                print('Training Data Eval:')
                batch_data = sess.run(reader.dequeue(1))
                do_eval(sess, eval_correct, inputs_placeholder, labels_placeholder, batch_data, batch_size, "training")
            """

if __name__ == '__main__':
    main()