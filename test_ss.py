from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import pickle
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import test_data

# import input_data
import models
from tensorflow.python.platform import gfile
import merge_res
FLAGS = None

use_cache = True

def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # sess = tf.Session(config=config)
    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=config)

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    label_count = len(FLAGS.wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        label_count,
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    names_file = 'out/names.pkl'
    res_files, words_list = merge_res.pickle_load(names_file)
    test_fgps = np.load('out/test_fgps.npy')
    # audio_processor = test_data.AudioProcessor(
    #     FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
    #     FLAGS.unknown_percentage,
    #     FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    #     FLAGS.testing_percentage, model_settings)
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)
    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    prob = tf.nn.softmax( logits )
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Loading from step: %d ', start_step)

    set_size = test_fgps.shape[0]
    tf.logging.info('set_size=%d', set_size)
    res_words = []
    test_probs = []

    def get_data(batch_size, i, model_settings):
        sample_count = max(0, min(batch_size, set_size- i))
        return test_fgps[i:i+sample_count]


    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints = get_data(FLAGS.batch_size, i, model_settings)
        test_prob, test_indices = sess.run([prob, predicted_indices],
                       feed_dict={
                           fingerprint_input: test_fingerprints,
                           dropout_prob: 1.0
                       }
                       )
        for indice in test_indices:
            res_words.append( words_list[indice] )
        test_probs.append(test_prob)
        if (i//FLAGS.batch_size) % 300 == 0:
            print('progress ', i, '/', set_size )
    tf.logging.info(res_words[0:10])
    test_probs = np.concatenate(test_probs, axis=0)
    print(test_probs.shape)
    suffix = '_'.join([word for word in wanted_words.split(',')])
    with open('out/{}_{}.pkl'.format(FLAGS.model_architecture, suffix), 'wb') as fout:
        pickle.dump(test_probs, fout)
    tf.logging.info('done.')

class ModelConfigureCNNBLSTM(object):
    #2层lstm，不使用conv与lstm的拼接
    def __init__(self, wanted_words):
        suffix = '_'.join( [word for word in wanted_words.split(',')] )
        self.summaries_dir='./logs_cnnblstm_ss_{}'.format(suffix)#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = './model_cnnblstm_ss_{}'.format(suffix)+'/cnnblstm.ckpt-15000'
        self.model_architecture='cnnblstm'  # 'conv',
        self.train_dir ='./model_cnnblstm_ss_{}'.format(suffix)

class ModelConfigureResnet(object):
    def __init__(self, wanted_words):
        suffix = '_'.join([word for word in wanted_words.split(',')])
        self.summaries_dir='./logs_resnet_ss_{}'.format(suffix)#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = './model_resnet_{}'.format(suffix)+'/resnet.ckpt-15000'
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_resnet_{}'.format(suffix)

if __name__ == '__main__':
    # wanted_words = 'up,off'  # 'go,down,no',#'stop,go'
    # wanted_words = 'go,down,no'
    # wanted_words = 'stop,go'
    # conf = ModelConfigureCNNBLSTM(wanted_words)
    wanted_words = 'stop,go'
    conf = ModelConfigureResnet(wanted_words)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        #already have data downloaded
        default='',#'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='test/',#'/tmp/speech_dataset/',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default=conf.lr_steps,
        help='How many training loops to run', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default=conf.learning_rate,
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default=conf.summaries_dir,
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default=wanted_words,#'yes,no,up,down,left,right,on,off,stop,go',
        # 'marvin,four,cat,on,dog,down,bed,two,go,wow,off,six,house,nine,eight,sheila,happy,'
        #         'five,tree,seven,stop,one,no,three,zero,up,yes,bird,right,left',
        #'yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=100,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default=conf.start_checkpoint,#'./model_train_resnet/resnet.ckpt-40000',#'./model_train2/conv.ckpt-18000',#'./model_train/conv.ckpt-18000',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default=conf.model_architecture,#'conv',
        help='What model architecture to use')
    parser.add_argument(
        '--train_dir',
        type=str,
        default=conf.train_dir,#'./model_train2',  # '/tmp/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,#100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
