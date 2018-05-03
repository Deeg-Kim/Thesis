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

from wavenet import AudioReader

def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))

coord = tf.train.Coordinator()
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

directory = './sampleTrue'
reader = AudioReader(directory, coord, sample_rate = 22000, gc_enabled=False, receptive_field=5117, sample_size=15117, silence_threshold=0.05)
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
reader.start_threads(sess)

name_in = "simplegenerate-60.wav"
name_out = "destandardized.wav"

in_data, _ = librosa.load(name_in, sr=22000)

data = sess.run(reader.dequeue(1))

data = np.array(data[0])
mean = np.mean(data)
std = np.std(data)

print(len(in_data))

samples = []
for i in range(11000):
	samples.append(float(in_data[i] * std) + float(mean))

waveform = np.reshape(samples, [11000])
write_wav(waveform, 44000, name_out)