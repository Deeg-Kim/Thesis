from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader

SAMPLES = 15117
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
SILENCE_THRESHOLD = 0.1

def get_generator_input_sampler():
    return lambda mu, sigma, n: np.random.normal(mu, sigma, size=n)

def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        '--samples',
        type=int,
        default=SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
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
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
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
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                              "not specified. Use --gc_id to specify global "
                              "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size,
                silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=args.gc_cardinality)

    gi_sampler = get_generator_input_sampler()

    # White noise generator params
    white_mean = 0
    white_sigma = 1
    white_length = 20234

    white_noise = gi_sampler(white_mean, white_sigma, white_length)

    loss = net.loss(input_batch=tf.convert_to_tensor(white_noise, dtype=np.float32), name='generator')

    samples = tf.placeholder(tf.int32)

    if args.fast_generation:
        next_sample = net.predict_proba_incremental(samples, args.gc_id)
    else:
        next_sample = net.predict_proba(samples, args.gc_id)

    if args.fast_generation:
        sess.run(tf.global_variables_initializer())
        sess.run(net.init_ops)

    decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

    quantization_channels = wavenet_params['quantization_channels']
    
    '''
    # Silence with a single random sample at the end.
    waveform = [quantization_channels / 2] * (net.receptive_field - 1)
    waveform.append(np.random.randint(quantization_channels))
    '''
    waveform = [0]

    last_sample_timestamp = datetime.now()
    for step in range(args.samples):
        if args.fast_generation:
            outputs = [next_sample]
            outputs.extend(net.push_ops)
            window = waveform[-1]
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform
            outputs = [next_sample]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]

        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged after
        # scaling.
        if args.temperature == 1.0:
            np.testing.assert_allclose(
                    prediction, scaled_prediction, atol=1e-5,
                    err_msg='Prediction scaling at temperature=1.0 '
                            'is not working as intended.')

        sample = np.random.choice(
            np.arange(quantization_channels), p=scaled_prediction)
        waveform.append(sample)

    # Introduce a newline to clear the carriage return from the progress.
    print()
    del waveform[0]
    print(waveform)
    print()

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
