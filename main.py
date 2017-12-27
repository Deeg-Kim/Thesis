# Code inspired by https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

import numpy as np

from wavenet import WaveNetModel, AudioReader, optimizer_factory

def get_generator_input_sampler():
	return lambda mu, sigma, n: np.random.normal(mu, sigma, size=n)

