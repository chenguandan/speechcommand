from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

FLAGS = None

from keras.backend.tensorflow_backend import set_session

feat_type = None #'normmfcc'#'dmfcc'#None  'normmfcc'
label_smooth = False
do_cache = False
model_boost_step = 0


def cache_data(sess, model_settings, audio_processor, time_shift_samples):
    set_size = audio_processor.set_size('training')
    print('train set size', set_size)
    # res_files = [get_file(tup['file']) for tup in audio_processor.data_index['training']]
    test_fgps = []
    labels = []
    for i in range(1000):#12000
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess, feat_type=feat_type,
            label_smooth=label_smooth)
        test_fgps.append( train_fingerprints )
        labels.append( train_ground_truth )
        if i % 100 == 0:
            print('progress ', i, '/', set_size )
    test_fgps = np.concatenate(test_fgps, axis=0)
    labels = np.concatenate( labels, axis =0 )
    print(test_fgps.shape)
    print(labels.shape)
    if feat_type is None:
        np.savez('out/train_fgps.npz',x=test_fgps,y=labels)
    else:
        np.savez('out/train_fgps_{}.npz'.format(feat_type), x=test_fgps, y = labels)


def get_graph( model_settings, is_last_model = False ):
    # Build a graph containing `net1`.
    with tf.Graph().as_default() as net1_graph:
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
        ws_input = tf.placeholder(
            tf.float32, [None, ], name='input_w')
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
            cross_entropy_mean_w = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=ground_truth_input, logits=logits) * ws_input)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            # train_step = tf.train.GradientDescentOptimizer(
            #     learning_rate_input).minimize(cross_entropy_mean)
            train_step = tf.train.AdamOptimizer(
                learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean_w)
        prob = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(logits, 1)
        expected_indices = tf.argmax(ground_truth_input, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # sess = tf.Session(config=config)
    if is_last_model:
        sess = tf.Session(graph=net1_graph, config=config)
    else:
        sess = tf.InteractiveSession(graph=net1_graph, config=config)
    sess.run(init_op)
    # saver.restore(sess, 'epoch_10.ckpt')
    if is_last_model:
        if FLAGS.start_checkpoint:
            # models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
            saver.restore(sess, FLAGS.start_checkpoint)
    return sess, saver, logits, dropout_prob, fingerprint_input, ground_truth_input,ws_input, learning_rate_input,\
        increment_global_step,confusion_matrix, evaluation_step, prob, predicted_indices,cross_entropy_mean,\
        train_step


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)


    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # sess = tf.Session(config=config)
    sess_global = tf.InteractiveSession(graph=tf.get_default_graph(), config=config)

    sess1, saver1, logits1, dropout_prob1, fingerprint_input1, ground_truth_input1, ws_input1, learning_rate_input1,\
    increment_global_step1, confusion_matrix1, evaluation_step1, prob1,predicted_indices1,cross_entropy_mean1,train_step1\
        = get_graph(model_settings, True)
    sess, saver, logits, dropout_prob, fingerprint_input, ground_truth_input, ws_input,learning_rate_input, \
    increment_global_step, confusion_matrix, evaluation_step, prob,predicted_indices,cross_entropy_mean, \
    train_step = get_graph(model_settings, False)


    start_step = 1
    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    if feat_type is None:
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
    else:
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + feat_type + '/train',
                                             sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + feat_type + '/validation')

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
            'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    best_loss = 10.0
    best_acc = 0.0
    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    # def get_data(batch_size, i, model_settings):
    #     i = i%set_size
    #     sample_count = max(0, min(batch_size, set_size- i))
    #     return train_xs[i:i+sample_count], train_ys[i:i+sample_count],inst_ws[i:i+sample_count]


    for training_step in xrange(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break
        # Pull the audio samples we'll use for training.
        # train_fingerprints, train_ground_truth = audio_processor.get_data(
        #     FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
        #     FLAGS.background_volume, time_shift_samples, 'training', sess, feat_type=feat_type, label_smooth=label_smooth)

        train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess_global, feat_type=feat_type, label_smooth=label_smooth)
        # Run the graph with this batch of training data.
        test_prob, test_indices = sess1.run([prob1, predicted_indices1],
                                           feed_dict={
                                               fingerprint_input1: train_fingerprints,
                                               dropout_prob1: 1.0
                                           }
                                           )
        ws = np.zeros((train_fingerprints.shape[0],))
        for ii in range(train_fingerprints.shape[0]):
            if test_indices[ii] == np.argmax(train_ground_truth[ii]):
                #TODO 上次的w乘以当前的
                ws[ii] = 1.0 * np.exp(-1.47)
            else:
                ws[ii] = 1.0 * np.exp(1.47)
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                ws_input: ws,
                learning_rate_input: learning_rate_value,
                dropout_prob: 1.0
            })

        train_writer.add_summary(train_summary, training_step)
        # if (training_step % FLAGS.eval_step_interval) == 0:
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))
        is_last_step = (training_step == training_steps_max)
        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            set_size = audio_processor.set_size('validation')
            total_accuracy = 0
            total_cross_entropy = 0.0
            total_conf_matrix = None
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                             0.0, 0, 'validation', sess_global,feat_type=feat_type))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, val_cross_entropy_value, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, cross_entropy_mean, confusion_matrix],
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 1.0,
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, set_size - i)
                total_accuracy += (validation_accuracy * batch_size) / set_size
                total_cross_entropy += (val_cross_entropy_value*batch_size)/set_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    if total_conf_matrix.shape[0] < conf_matrix.shape[0]:
                        total_conf_matrix = conf_matrix
                    elif total_conf_matrix.shape[0] > conf_matrix.shape[0]:
                        total_conf_matrix[:conf_matrix.shape[0],:conf_matrix.shape[1]] += conf_matrix[:,:]
                    else:
                        total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, total_accuracy * 100, set_size))
            if total_accuracy > best_acc:
                best_acc = total_accuracy
            if total_cross_entropy < best_loss:
                best_loss = total_cross_entropy
        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            if feat_type is None:
                checkpoint_path = os.path.join(FLAGS.train_dir,
                                               FLAGS.model_architecture + '.ckpt')
            else:
                checkpoint_path = os.path.join(FLAGS.train_dir+feat_type,
                                               FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    print('best loss', best_loss, 'best acc', best_acc)
    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None

    for i in xrange(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess_global,feat_type=feat_type)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
                dropout_prob: 1.0
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            if total_conf_matrix.shape[0] < conf_matrix.shape[0]:
                total_conf_matrix = conf_matrix
            elif total_conf_matrix.shape[0] > conf_matrix.shape[0]:
                total_conf_matrix[:conf_matrix.shape[0], :conf_matrix.shape[1]] += conf_matrix[:, :]
            else:
                total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,
                                                             set_size))
class ModelConfigure(object):
    def __init__(self):
        self.summaries_dir='./retrain_resnet2_logs'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = './model_train_resnet2/resnet.ckpt-11000'  # './model_train_resnet/resnet.ckpt-40000',#'./model_train2/resnet.ckpt-27600',
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_train_resnet2'
            # '/tmp/speech_commands_train',
            # cnn   './model_train'
            # resnet dropout 0.5 model_train2
            # resnet dropout 1.0 model_train_resnet

class ModelConfigureResnet(object):
    def __init__(self):
        self.summaries_dir='./logs_resnet'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_resnet'
            # '/tmp/speech_commands_train',
            # cnn   './model_train'
            # resnet dropout 0.5 model_train2
            # resnet dropout 1.0 model_train_resnet

class ModelConfigureResnetLS(object):
    def __init__(self):
        self.summaries_dir='./logs_resnetls'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_resnetls'
            # '/tmp/speech_commands_train',
            # cnn   './model_train'
            # resnet dropout 0.5 model_train2
            # resnet dropout 1.0 model_train_resnet

class ModelConfigureResnet101(object):
    def __init__(self):
        self.summaries_dir='./logs_resnet101'#retrain_logs
        self.lr_steps = '8000,8000,6000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 32
        self.start_checkpoint = ''
        self.model_architecture='resnet101'  # 'conv',
        self.train_dir ='./model_resnet101'

class ModelConfigureResnetblstm(object):
    def __init__(self):
        self.summaries_dir='./logs_resnetblstm_step{}'.format(model_boost_step)
        self.lr_steps = '3000,2000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        if model_boost_step == 0:
            self.start_checkpoint = 'model_resnetblstm/resnetblstm.ckpt-11000'
        else:
            self.start_checkpoint = 'model_resnetblstm_step{}/resnetblstm.ckpt-5000'.format(model_boost_step)
        self.model_architecture='resnetblstm'  # 'conv',
        self.train_dir ='./model_resnetblstm_step{}'.format(model_boost_step)


class ModelConfigureXception(object):
    def __init__(self):
        self.summaries_dir='./logs_xception'#retrain_logs
        self.lr_steps = '8000,8000,6000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 32
        self.start_checkpoint = ''
        self.model_architecture='xception'  # 'conv',
        self.train_dir ='./model_xception'

class ModelConfigureCNNBLSTM(object):
    #2层lstm，不使用conv与lstm的拼接
    def __init__(self):
        # self.summaries_dir='./logs_cnnblstm'#retrain_logs
        # self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        # self.learning_rate = '0.001,0.0003,0.0001'
        # self.batch_size = 64
        # self.start_checkpoint = ''
        # self.model_architecture='cnnblstm'  # 'conv',
        # self.train_dir ='./model_cnnblstm'
        self.summaries_dir = './logs_cnnblstm_step{}'.format(model_boost_step)  # retrain_logs
        self.lr_steps = '6000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        if model_boost_step == 0:
            self.start_checkpoint = 'model_cnnblstm/cnnblstm.ckpt-15000'
        else:
            self.start_checkpoint = 'model_cnnblstm_step{}/cnnblstm.ckpt-9000'.format(model_boost_step - 1)

        self.model_architecture = 'cnnblstm'  # 'conv',
        self.train_dir = './model_cnnblstm_step{}'.format(model_boost_step)

class ModelConfigureCNNBLSTM2(object):
    #2层lstm，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm2'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm2'  # 'conv',
        self.train_dir ='./model_cnnblstm2'

class ModelConfigureCNNBLSTM6(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm6'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm6'  # 'conv',
        self.train_dir ='./model_cnnblstm6'


class ModelConfigureCNNBLSTMBND(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstmbnd'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstmbnd'  # 'conv',
        self.train_dir ='./model_cnnblstmbnd'

class ModelConfigureLace(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace_step{}'.format(model_boost_step)#retrain_logs
        self.lr_steps = '6000,2000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        if model_boost_step == 0:
            self.start_checkpoint = 'model_lace/lace.ckpt-12000'
        else:
            self.start_checkpoint = 'model_lace_step{}/lace.ckpt-8000'.format(model_boost_step-1)

        self.model_architecture='lace'  # 'conv',
        self.train_dir ='./model_lace_step{}'.format(model_boost_step)


class ModelConfigureLace2(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace2'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='lace2'  # 'conv',
        self.train_dir ='./model_lace2'

if __name__ == '__main__':
    if label_smooth:
        conf = ModelConfigureResnetLS()
    else:
        # conf = ModelConfigureResnet()
        # conf = ModelConfigureResnetblstm()
        # conf = ModelConfigureResnet101()
        # conf = ModelConfigureXception()
        # conf = ModelConfigureCNNBLSTMBND()
        # conf = ModelConfigureCNNBLSTM()
        # conf = ModelConfigureCNNBLSTM2()
        # conf = ModelConfigureCNNBLSTM6()
        conf = ModelConfigureLace( )
        # conf = ModelConfigureLace2()
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
        default='train/audio/',#'/tmp/speech_dataset/',
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
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default=conf.lr_steps,  # 15000,10000,3000
        # 15000,3000，但是原本batch_size是100，所以这里要改steps为3倍，使用Adam以后可以适当减小
        #使用Adam以后，3000差不多就已经快收敛了，
        help='How many training loops to run', )
    parser.add_argument(
        '--learning_rate',
        type=str,
        default=conf.learning_rate,
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=conf.batch_size,#64,#100,
        help='How many items to train with at once', )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default=conf.summaries_dir,#retrain_logs
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,#up,off最容易混； on,right与unk； go与down，go与no(test)，down与no（test),up与no(test)
        #stop与go(test)，off与up(test)
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
        default=conf.start_checkpoint,#'./model_train_resnet/resnet.ckpt-40000',#'./model_train2/resnet.ckpt-27600',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default=conf.model_architecture,#'conv',
        help='What model architecture to use')
    parser.add_argument(
        '--train_dir',
        type=str,
        default=conf.train_dir,
        # '/tmp/speech_commands_train',
        # cnn   './model_train'
        #resnet dropout 0.5 model_train2
        #resnet dropout 1.0 model_train_resnet
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#input shape: (98,40)
#In TensorFlow audio recognition tutorial, no normalize the MFCCs before as input,
    #  I wonder if it's necessary to do normalize ?