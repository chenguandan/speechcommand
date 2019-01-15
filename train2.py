"""
加入了i-vector（person vector)
没有明显的效果提升
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data2
import models2
from tensorflow.python.platform import gfile

FLAGS = None

use_keras = False
import keras
from keras.backend.tensorflow_backend import set_session

def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # sess = tf.Session(config=config)
    # Start a new TensorFlow session.
    sess = tf.InteractiveSession(config=config)
    set_session(sess)

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models2.prepare_model_settings(
        len(input_data2.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    audio_processor = input_data2.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    person_count = len(audio_processor.person_dict)
    model_settings['person_count'] = person_count
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits, person_logits, dropout_prob = models2.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        is_training=True)

    # Define loss and optimizer
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')
    ground_truth_person_input = tf.placeholder(
        tf.float32, [None, person_count], name='groundtruth_person_input')

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
        cross_entropy_mean_person = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_person_input, logits=person_logits))



    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        # train_step = tf.train.GradientDescentOptimizer(
        #     learning_rate_input).minimize(cross_entropy_mean)
        train_step = tf.train.AdamOptimizer(
            learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean)
        train_step_person = tf.train.AdamOptimizer(
            learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean_person)
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    predicted_indices_person = tf.argmax(person_logits, 1)
    expected_indices_person = tf.argmax(ground_truth_person_input, 1)
    correct_prediction_person = tf.equal(predicted_indices_person, expected_indices_person)
    evaluation_step_person = tf.reduce_mean(tf.cast(correct_prediction_person, tf.float32))


    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)


    with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
        learning_rate = 0.001
        k = 19.0
        #earning_rate / (1 + decay_rate * global_step / decay_step)
        learning_rate_d = tf.train.inverse_time_decay(learning_rate, global_step, 10000, k)
        train_step_d = (
            tf.train.GradientDescentOptimizer(learning_rate_d)
                .minimize(cross_entropy_mean, global_step = global_step)
        )
        train_step_person_d = (
            tf.train.GradientDescentOptimizer(learning_rate_d)
                .minimize(cross_entropy_mean_person, global_step=global_step)
        )

    saver = tf.train.Saver(tf.global_variables())
    best_acc = 0.0
    best_acc_saver = tf.train.Saver(tf.global_variables())
    best_loss = 10.0
    best_loss_saver = tf.train.Saver(tf.global_variables())
    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    # merged_summaries = tf.summary.merge_all()
    merged_summaries = tf.summary.merge([tf.summary.scalar('cross_entropy', cross_entropy_mean),
        tf.summary.scalar('accuracy', evaluation_step)])
    merged_summaries_person = tf.summary.merge([tf.summary.scalar('cross_entropy_person', cross_entropy_mean_person),
                                                tf.summary.scalar('accuracy_person', evaluation_step_person),
                                                ])

    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1
    fixed = False
    if fixed:
        params = [param for param in tf.global_variables() if
                  ('dense_fn' not in param.name and 'global_step' not in param.name)]
        saver = tf.train.Saver(params)
        saver.restore(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)
    else:
        if FLAGS.start_checkpoint:
            models2.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
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
        if int(training_steps_max*0.8)<training_step:
            train_step = train_step_d
            train_step_person = train_step_person_d
        # Pull the audio samples we'll use for training.
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)
        # Run the graph with this batch of training data.
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: 1.0
            })

        train_fingerprints_p, train_ground_truth_p = audio_processor.get_ivector_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)
        train_summaryp, train_accuracyp, cross_entropy_valuep, _ = sess.run(
            [
                merged_summaries_person, evaluation_step_person, cross_entropy_mean_person, train_step_person
            ],
            feed_dict={
                fingerprint_input: train_fingerprints_p,
                ground_truth_person_input: train_ground_truth_p,
                learning_rate_input: learning_rate_value,
                dropout_prob: 1.0
            })
        train_writer.add_summary(train_summary, training_step)
        train_writer.add_summary(train_summaryp, training_step)
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
                                             0.0, 0, 'validation', sess))
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

            set_size = audio_processor.person_set_size('validation')
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprintsp, validation_ground_truthp = (
                    audio_processor.get_ivector_data(FLAGS.batch_size, i, model_settings, 0.0,
                                                     0.0, 0, 'validation', sess))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summaryp, validation_accuracyp, val_cross_entropy_valuep = sess.run(
                    [merged_summaries_person, evaluation_step_person, cross_entropy_mean_person],
                    feed_dict={
                        fingerprint_input: validation_fingerprintsp,
                        ground_truth_person_input: validation_ground_truthp,
                        dropout_prob: 1.0,
                    })
                validation_writer.add_summary(validation_summaryp, training_step)

            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, total_accuracy * 100, set_size))
            # summary = tf.Summary()
            # summary.value.add(tag="smooth_acc" , simple_value=total_accuracy)
            # validation_writer.add_summary(summary, global_step)
            if total_accuracy > best_acc:
                best_acc = total_accuracy
                best_acc_saver.save(sess, os.path.join(FLAGS.train_dir, 'best_acc.ckpt'))
            if total_cross_entropy < best_loss:
                best_loss = total_cross_entropy
                best_loss_saver.save(sess, os.path.join(FLAGS.train_dir, 'best_loss.ckpt'))
        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            checkpoint_path = os.path.join(FLAGS.train_dir,
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
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        if use_keras:
            test_accuracy, conf_matrix = sess.run(
                [evaluation_step, confusion_matrix],
                feed_dict={
                    fingerprint_input: test_fingerprints,
                    ground_truth_input: test_ground_truth,
                    keras.backend.learning_phase():1
                })
        else:
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
        self.start_checkpoint = './model_train_resnet2/resnet.ckpt-11000'  # './model_train_resnet/resnet.ckpt-40000',#'./model_train2/resnet.ckpt-27600',
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_train_resnet2'
            # '/tmp/speech_commands_train',
            # cnn   './model_train'
            # resnet dropout 0.5 model_train2
            # resnet dropout 1.0 model_train_resnet

class ModelConfigureResnetblstm(object):
    #2层lstm，不使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_resnetblstmp'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='resnetblstmp'  # 'conv',
        self.train_dir ='./model_resnetblstmp'

class ModelConfigureCNNBLSTM(object):
    #2层lstm，不使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstmp'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='cnnblstmp'  # 'conv',
        self.train_dir ='./model_cnnblstmp'

class ModelConfigureCNNBLSTM2(object):
    #2层lstm，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm2'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm2'  # 'conv',
        self.train_dir ='./model_cnnblstm2'

class ModelConfigureCNNBLSTM6(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstm6'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='cnnblstm6'  # 'conv',
        self.train_dir ='./model_cnnblstm6'


class ModelConfigureCNNBLSTMBND(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_cnnblstmbnd'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='cnnblstmbnd'  # 'conv',
        self.train_dir ='./model_cnnblstmbnd'

class ModelConfigureLace(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace'#retrain_logs
        self.lr_steps = '4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='lace'  # 'conv',
        self.train_dir ='./model_lace'


class ModelConfigureLace2(object):
    #6层的gru，使用conv与lstm的拼接
    def __init__(self):
        self.summaries_dir='./logs_lace2'#retrain_logs
        self.lr_steps = '4000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0005,0.0001'
        self.start_checkpoint = ''
        self.model_architecture='lace2'  # 'conv',
        self.train_dir ='./model_lace2'

if __name__ == '__main__':
    # conf = ModelConfigureLace2()
    # conf = ModelConfigureCNNBLSTMBND()
    # conf = ModelConfigureCNNBLSTM()
    conf = ModelConfigureResnetblstm()
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
        default=64,#100,
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

#num person: 1887