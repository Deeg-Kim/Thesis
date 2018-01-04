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

def fill_feed_dict(sess, coord, inputs_pl, labels_pl, batch_size):
    # Feed dict for placeholders from placeholder_inputs()
    directory = './sample1'

    reader = AudioReader(directory, coord, sample_rate = 16000, gc_enabled=False, receptive_field=5117, sample_size=10000, silence_threshold=0.1)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    inputs_feed = []
    labels_feed = np.ones((batch_size), dtype=np.int32)

    for batch in range(batch_size):
        batch_data = sess.run(reader.dequeue(1))
        list_data = convert_dequeue_to_list(batch_data)

        inputs_feed.append(list_data)

    inputs_feed = np.array(inputs_feed, dtype=np.float32)

    feed_dict = {
        inputs_pl: inputs_feed,
        labels_pl: labels_feed
    }

    return feed_dict

def main():

    with tf.Graph().as_default():
        coord = tf.train.Coordinator()
        sess = tf.Session()

        # data = reader.dequeue(1)
        # data = sess.run(data)

        batch_size = 100
        hidden1_units = 7884
        hidden2_units = 5256
        hidden3_units = 2628
        max_steps = 100
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

        for step in range(max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(sess, coord, inputs_placeholder, labels_placeholder, batch_size)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            print('Step %d: loss = %.7f (%.3f sec)' % (step, loss_value, duration))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            if (step + 1) % 10 == 0 or (step + 1) == max_steps:
                checkpoint_file = os.path.join('./logdir/init-train/', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

if __name__ == '__main__':
    main()