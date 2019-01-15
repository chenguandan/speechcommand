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

FLAGS = None

do_cache = True
use_cache = False

feat_type = 'dmfcc'#None  'normmfcc'

def cache_data( sess, model_settings, audio_processor):
    # model_settings = models.prepare_model_settings(
    #     len(test_data.prepare_words_list(FLAGS.wanted_words.split(','))),
    #     FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
    #     FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    # audio_processor = test_data.AudioProcessor(
    #     FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
    #     FLAGS.unknown_percentage,
    #     FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    #     FLAGS.testing_percentage, model_settings)
    set_size = audio_processor.set_size('testing')
    def get_file(file):
        index = file.rfind('/')
        index2 = file.rfind('\\')
        if index < 0:
            index = index2
        subfile = file[index + 1:]
        return subfile

    res_files = [get_file(tup['file']) for tup in audio_processor.data_index['testing']]
    test_fgps = []
    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, _ = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess, feat_type=feat_type)
        test_fgps.append( test_fingerprints )
        if (i//FLAGS.batch_size) % 1000 == 0:
            print('progress ', i, '/', set_size )
    test_fgps = np.concatenate(test_fgps, axis=0)
    print(test_fgps.shape)
    res_words = []
    #TODO labels
    # with open('out/test_fgps.pkl','wb') as fout:
    #     pickle.dump(test_fgps, fout)
    if feat_type is None:
        np.save('out/test_fgps.npy',test_fgps)
        with open('out/names.pkl', 'wb') as fout:
            pickle.dump( (res_files, audio_processor.words_list), fout)
    else:
        np.save('out/test_fgps_{}.npy'.format(feat_type), test_fgps)
        with open('out/names_{}.pkl'.format(feat_type), 'wb') as fout:
            pickle.dump((res_files, audio_processor.words_list), fout)

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
    model_settings = models.prepare_model_settings(
        len(test_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    audio_processor = test_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    if do_cache:
        cache_data(sess, model_settings, audio_processor)
        return
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
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

    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    res_words = []
    def get_file(file):
        index = file.rfind('/')
        index2 = file.rfind('\\')
        if index<0:
            index = index2
        subfile = file[index+1:]
        return subfile
    res_files = [get_file( tup['file']) for tup in audio_processor.data_index['testing']]
    test_probs = []
    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        test_prob, test_indices = sess.run([prob, predicted_indices],
                       feed_dict={
                           fingerprint_input: test_fingerprints,
                           dropout_prob: 1.0
                       }
                       )
        #(32,)
        # print(test_indices.shape)
        for indice in test_indices:
            res_words.append( audio_processor.words_list[indice] )
        test_probs.append(test_prob)

        # test_accuracy, conf_matrix = sess.run(
        #     [evaluation_step, confusion_matrix],
        #     feed_dict={
        #         fingerprint_input: test_fingerprints,
        #         ground_truth_input: test_ground_truth,
        #         dropout_prob: 1.0
        #     })
        batch_size = min(FLAGS.batch_size, set_size - i)
    tf.logging.info(res_words[0:10])
    test_probs = np.concatenate(test_probs, axis=0)
    print(test_probs.shape)
    with open('out/{}.pkl'.format(FLAGS.model_architecture), 'wb') as fout:
        pickle.dump(test_probs, fout)
    with open('res.csv', 'w') as fout:
        fout.write('fname,label\n')
        if len(res_words)> len(res_files):
            res_files = res_files[:len(res_words)]
        for f, word in zip(res_files, res_words):
            real_word = word
            if word == test_data.SILENCE_LABEL:
                real_word = 'silence'
            elif word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'unknown'
            if real_word == test_data.SILENCE_LABEL or real_word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'silence'
            fout.write('{},{}\n'.format(f,real_word))
    tf.logging.info('done.')

class ModelConfigureResnet2(object):
    def __init__(self):
        self.summaries_dir='./retrain_resnet2_logs'#retrain_logs
        self.lr_steps = '8000,3000' # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = './model_train_resnet2/resnet.ckpt-11000'  # './model_train_resnet/resnet.ckpt-40000',#'./model_train2/resnet.ckpt-27600',
        self.model_architecture='resnet' # 'conv',
        self.train_dir ='./model_train_resnet2'

class ModelConfigureCNNBLSTM(object):
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.start_checkpoint = './model_cnnblstm/cnnblstm.ckpt-14600'
        self.model_architecture='cnnblstm' # 'conv',
        self.train_dir ='./model_cnnblstm'

class ModelConfigureCNNBLSTM6(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm6'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = './model_cnnblstm6/best_acc.ckpt'#cnnblstm6.ckpt-7000
        self.model_architecture='cnnblstm6'  # 'conv',
        self.train_dir ='./model_cnnblstm6'

class ModelConfigureLace(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = './model_lace/lace.ckpt-7000'
        self.model_architecture='lace'  # 'conv',
        self.train_dir ='./model_lace'

class ModelConfigureCNNBLSTMBND(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstmbnd'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.start_checkpoint = './model_cnnblstmbnd/best_loss.ckpt'
        self.model_architecture='cnnblstmbnd'  # 'conv',
        self.train_dir ='./model_cnnblstmbnd'


if __name__ == '__main__':
    # conf = ModelConfigureResnet2()
    # conf = ModelConfigureCNNBLSTM6()
    conf = ModelConfigureLace()
    # conf = ModelConfigureCNNBLSTMBND()
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
        default='yes,no,up,down,left,right,on,off,stop,go',
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
