from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
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

def fill_feed_dict(batch_data, inputs_pl, labels_pl, batch_size):
    # Feed dict for placeholders from placeholder_inputs()
    inputs_feed = []
    labels_feed = np.ones((batch_size), dtype=np.int32)

    for batch in range(batch_size):
        list_data = convert_dequeue_to_list(batch_data)

        inputs_feed.append(list_data)

    inputs_feed = np.array(inputs_feed, dtype=np.float32)

    feed_dict = {
        inputs_pl: inputs_feed,
        labels_pl: labels_feed
    }

    return feed_dict

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

        batch_size = 100
        hidden1_units = 7884
        hidden2_units = 5256
        hidden3_units = 2628
        max_steps = 1000
        learning_rate = 1e-3

        inputs_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        logits = ffnn.inference(inputs_placeholder, hidden1_units, hidden2_units, hidden3_units)
        loss = ffnn.loss(logits, labels_placeholder)
        train_op = ffnn.training(loss, learning_rate)
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

        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        directory = './sample1'
        reader = AudioReader(directory, coord, sample_rate = 16000, gc_enabled=False, receptive_field=5117, sample_size=10000, silence_threshold=0.1)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)

        for step in range(saved_global_step + 1, max_steps):
            start_time = time.time()

            batch_data = sess.run(reader.dequeue(1))
            feed_dict = fill_feed_dict(batch_data, inputs_placeholder, labels_placeholder, batch_size)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            print('Step %d: loss = %.7f (%.3f sec)' % (step, loss_value, duration))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            if (step + 1) % 100 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join('./logdir/init-train/', 'model.ckpt')
                print("Saving!")
                saver.save(sess, checkpoint_file, global_step=step)

if __name__ == '__main__':
    main()