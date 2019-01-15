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
import models_pair
import models_p as models

from tensorflow.python.platform import gfile

FLAGS = None

use_keras = False
import keras
from keras.backend.tensorflow_backend import set_session

is_person = False
feat_type = None #'normmfcc'#'dmfcc'#None  'normmfcc'
label_smooth = False

def load_graph(model_settings, is_person = False):
    with tf.Graph().as_default() as net1_graph:
        fingerprint_size = model_settings['fingerprint_size']
        fingerprint_input1 = tf.placeholder(
            tf.float32, [None, fingerprint_size], name='fingerprint_input1')
        fingerprint_input2 = tf.placeholder(
            tf.float32, [None, fingerprint_size], name='fingerprint_input2')
        embed1, embed2, logits, dropout_prob = models_pair.create_model(
            fingerprint_input1,
            fingerprint_input2,
            model_settings,
            'lace',
            is_training=True)
        # Define loss and optimizer
        ground_truth_input = tf.placeholder(
            tf.float32, [None, 1], name='groundtruth_input')
        # Optionally we can add runtime checks to spot when NaNs or other symptoms of
        # numerical errors start occurring during training.
        control_dependencies = []
        if FLAGS.check_nans:
            checks = tf.add_check_numerics_ops()
            control_dependencies = [checks]
        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=ground_truth_input, logits=logits))
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            train_step = tf.train.AdamOptimizer(
                learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean)
        predicted_indices = tf.cast(tf.greater_equal(logits, 0.5), tf.float32)
        expected_indices = ground_truth_input
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', evaluation_step)
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # sess = tf.Session(config=config)
    sess = tf.Session(graph=net1_graph, config=config)
    sess.run(init_op)
    start_step = 1
    start_checkpoint = 'model_lace_p{}/lace.ckpt-11000'.format(is_person)
    saver.restore(sess, start_checkpoint)
    return sess, fingerprint_input1, embed1, dropout_prob


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # sess = tf.Session(config=config)
    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=config)
    set_session(sess)

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
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    sess_w, fingerprint_input1_w, embed1_w, dropout_prob_w =load_graph(model_settings, is_person = False)
    sess_p, fingerprint_input1_p, embed1_p, dropout_prob_p = load_graph(model_settings, is_person=True)
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')
    person_embed_input = tf.placeholder(
        tf.float32, [None, 512], name='person_embed')
    word_embed_input = tf.placeholder(
        tf.float32, [None, 512], name='word_embed')
    logits, dropout_prob = models.create_model(
        fingerprint_input,
        person_embed_input,
        word_embed_input,
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
        # train_step = tf.train.GradientDescentOptimizer(
        #     learning_rate_input).minimize(cross_entropy_mean)
        train_step = tf.train.AdamOptimizer(
            learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean)
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())
    best_acc = 0.0
    best_acc_saver = tf.train.Saver(tf.global_variables())
    best_loss = 10.0
    best_loss_saver = tf.train.Saver(tf.global_variables())
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
    tf.global_variables_initializer().run()

    start_step = 1
    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # Save list of words.
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
            'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in xrange(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break
        # Pull the audio samples we'll use for training.
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess, feat_type=feat_type,
            label_smooth=label_smooth)
        # Run the graph with this batch of training data.
        # sess_w, fingerprint_input1_w, embed1_w, dropout_prob_w
        word_embed = sess_w.run(embed1_w, feed_dict={
            fingerprint_input1_w:train_fingerprints,
            dropout_prob_w:1.0
        })
        person_embed = sess_p.run( embed1_p, feed_dict={
            fingerprint_input1_p:train_fingerprints,
            dropout_prob_p:1.0
        })
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                person_embed_input:person_embed,
                word_embed_input:word_embed,
                ground_truth_input: train_ground_truth,
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
                                             0.0, 0, 'validation', sess, feat_type=feat_type))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                word_embed = sess_w.run(embed1_w, feed_dict={
                    fingerprint_input1_w: validation_fingerprints,
                    dropout_prob_w: 1.0
                })
                person_embed = sess_p.run(embed1_p, feed_dict={
                    fingerprint_input1_p: validation_fingerprints,
                    dropout_prob_p: 1.0
                })
                validation_summary, validation_accuracy, val_cross_entropy_value, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, cross_entropy_mean, confusion_matrix],
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        person_embed_input:person_embed,
                        word_embed_input:word_embed,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 1.0,
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, set_size - i)
                total_accuracy += (validation_accuracy * batch_size) / set_size
                total_cross_entropy += (val_cross_entropy_value * batch_size) / set_size
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
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, total_accuracy * 100, set_size))
            if total_accuracy > best_acc:
                best_acc = total_accuracy
                best_acc_saver.save(sess, os.path.join(FLAGS.train_dir, 'best_acc.ckpt'))
            if total_cross_entropy < best_loss:
                best_loss = total_cross_entropy
                best_loss_saver.save(sess, os.path.join(FLAGS.train_dir, 'best_loss.ckpt'))
        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            if feat_type is None:
                checkpoint_path = os.path.join(FLAGS.train_dir,
                                               FLAGS.model_architecture + '.ckpt')
            else:
                checkpoint_path = os.path.join(FLAGS.train_dir + feat_type,
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
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess, feat_type=feat_type)
        word_embed = sess_w.run(embed1_w, feed_dict={
            fingerprint_input1_w: test_fingerprints,
            dropout_prob_w: 1.0
        })
        person_embed = sess_p.run(embed1_p, feed_dict={
            fingerprint_input1_p: test_fingerprints,
            dropout_prob_p: 1.0
        })
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                person_embed_input:person_embed,
                word_embed_input:word_embed,
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
        self.summaries_dir='./logs_resnet_p'
        self.lr_steps = '8000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_resnet_p'

class ModelConfigureResnet101(object):
    def __init__(self):
        self.summaries_dir='./logs_resnet101_p'#retrain_logs
        self.lr_steps = '8000,8000,6000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 32
        self.start_checkpoint = ''
        self.model_architecture='resnet101'  # 'conv',
        self.train_dir ='./model_resnet101_p'

class ModelConfigureResnetblstm(object):
    def __init__(self):
        self.summaries_dir='./logs_resnetblstm_p'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='resnetblstm'  # 'conv',
        self.train_dir ='./model_resnetblstm_p'


class ModelConfigureXception(object):
    def __init__(self):
        self.summaries_dir='./logs_xception_p'#retrain_logs
        self.lr_steps = '8000,8000,6000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 32
        self.start_checkpoint = ''
        self.model_architecture='xception'  # 'conv',
        self.train_dir ='./model_xception_p'

class ModelConfigureCNNBLSTM(object):
    #2层lstm，不使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm_p'
        self.lr_steps = '8000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm'  # 'conv',
        self.train_dir ='./model_cnnblstm_p'

class ModelConfigureCNNBLSTM2(object):
    #2层lstm，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm2_p'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm2'  # 'conv',
        self.train_dir ='./model_cnnblstm2_p'

class ModelConfigureCNNBLSTM6(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm6_p'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm6'  # 'conv',
        self.train_dir ='./model_cnnblstm6_p'


class ModelConfigureCNNBLSTMBND(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstmbnd_p'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='cnnblstmbnd'  # 'conv',
        self.train_dir ='./model_cnnblstmbnd_p'

class ModelConfigureLace(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace_p'
        self.lr_steps = '8000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='lace'  # 'conv',
        self.train_dir ='./model_lace_p'


class ModelConfigureLace2(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace2_p'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='lace2'  # 'conv',
        self.train_dir ='./model_lace2_p'

class ModelConfigureSEResnet(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_se_resnet_p'#retrain_logs
        self.lr_steps = '16000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 16
        self.start_checkpoint = ''
        self.model_architecture='se_resnet'  # 'conv',
        self.train_dir ='./model_se_resnet_p'

class ModelConfigureDpnet92(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_dpnet92'#retrain_logs
        self.lr_steps = '12000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = './model_dpnet92/dpnet92.ckpt-11000'
        self.model_architecture='dpnet92'  # 'conv',
        self.train_dir ='./model_dpnet92'


class ModelConfigureDpnet98(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_dpnet98'#retrain_logs
        self.lr_steps = '12000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='dpnet98'  # 'conv',
        self.train_dir ='./model_dpnet98'

class ModelConfigureDpnet107(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_dpnet107'#retrain_logs
        self.lr_steps = '8000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='dpnet107'  # 'conv',
        self.train_dir ='./model_dpnet107'

class ModelConfigureDpnet137(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_dpnet137'#retrain_logs
        self.lr_steps = '8000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='dpnet137'  # 'conv',
        self.train_dir ='./model_dpnet137'

if __name__ == '__main__':
    confs = []
    # conf = ModelConfigureResnet()
    # confs.append(conf)
    conf = ModelConfigureSEResnet() #
    confs.append(conf)
    conf = ModelConfigureResnetblstm()
    confs.append(conf)
    conf = ModelConfigureResnet101()
    confs.append(conf)
    conf = ModelConfigureXception()
    confs.append(conf)
    conf = ModelConfigureCNNBLSTMBND()
    confs.append(conf)
    conf = ModelConfigureCNNBLSTM() #
    confs.append(conf)
    conf = ModelConfigureCNNBLSTM2()
    confs.append(conf)
    conf = ModelConfigureCNNBLSTM6()
    confs.append(conf)
    conf = ModelConfigureLace( ) #
    confs.append(conf)
    conf = ModelConfigureLace2()
    confs.append(conf)
    # conf = ModelConfigureDpnet92()
    for conf in confs:
        tf.reset_default_graph()
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
        # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        main([sys.argv[0]] + unparsed)
#input shape: (98,40)
#In TensorFlow audio recognition tutorial, no normalize the MFCCs before as input,
    #  I wonder if it's necessary to do normalize ?