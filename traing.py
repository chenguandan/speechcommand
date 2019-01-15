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
import os
import numpy as np
from dcgan import *
from util import *
# from load import mnist_with_valid_set

n_epochs = 100
learning_rate = 0.0002#0.0002太小了？
batch_size = 64
image_shape = [98,40,1]
dim_z = 100
dim_W1 = 512
dim_W2 = 128
dim_W3 = 64
dim_channel = 1

visualize_dim=196
log_every = 20#100
step = 200


def train_dcgan():
    dcgan_model = DCGAN(
            batch_size=batch_size,
            image_shape=image_shape,
            dim_z=dim_z,
            dim_W1=dim_W1,
            dim_W2=dim_W2,
            dim_W3=dim_W3,
            )

    Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=10)

    discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    discrim_vars = [i for i in discrim_vars]
    gen_vars = [i for i in gen_vars]

    train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
    train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

    # Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

    tf.global_variables_initializer().run()

    Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim,dim_z))
    Y_np_sample = OneHot( np.random.randint(10, size=[visualize_dim]))
    iterations = 0
    k = 2
    log_every = 100
    step = 200

    for epoch in range(n_epochs):
        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]

        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
                ):

            Xs = trX[start:end].reshape( [-1, 28, 28, 1]) / 255.
            Ys = OneHot(trY[start:end])
            Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

            if np.mod( iterations, k ) != 0:
                _, gen_loss_val = sess.run(
                        [train_op_gen, g_cost_tf],
                        feed_dict={
                            Z_tf:Zs,
                            Y_tf:Ys
                            })
                discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf,p_real,p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
                if iterations % log_every == 0:
                    # print("=========== updating G ==========")
                    print("epoch:{}\tupdating G\titeration:{}\t gen loss:{}\t discrim loss:{} ", epoch,
                          iterations, gen_loss_val, discrim_loss_val)
                    # print("iteration:", iterations)
                    # print("gen loss:", gen_loss_val)
                    # print("discrim loss:", discrim_loss_val)

            else:
                _, discrim_loss_val = sess.run(
                        [train_op_discrim, d_cost_tf],
                        feed_dict={
                            Z_tf:Zs,
                            Y_tf:Ys,
                            image_tf:Xs
                            })
                gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen], feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})
                if iterations % log_every == 0:
                    # print("=========== updating D ==========")
                    print("epoch:{}\tupdating D\titeration:{}\t gen loss:{}\t discrim loss:{} ",
                          epoch, iterations,gen_loss_val, discrim_loss_val)
                    # print("gen loss:", gen_loss_val)
                    # print("discrim loss:", discrim_loss_val)
            if iterations % log_every == 0:
                print("Average P(real)={}\tAverage P(gen)={}", p_real_val.mean(), p_gen_val.mean() )
                # print("Average P(gen)=", p_gen_val.mean())

            # if np.mod(iterations, step) == 0:
            #     generated_samples = sess.run(
            #             image_tf_sample,
            #             feed_dict={
            #                 Z_tf_sample:Z_np_sample,
            #                 Y_tf_sample:Y_np_sample
            #                 })
            #     generated_samples = (generated_samples + 1.)/2.
            #     save_visualization(generated_samples, (14,14), save_path='./vis/sample_%04d.jpg' % int(iterations/step))

            iterations += 1


FLAGS = None

use_keras = False
from keras.backend.tensorflow_backend import set_session

def main(_):
    # We want to see all the logging messages for this tutorial.
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

    # fingerprint_input = tf.placeholder(
    #     tf.float32, [None, fingerprint_size], name='fingerprint_input')
    #
    # logits, dropout_prob = models.create_model(
    #     fingerprint_input,
    #     model_settings,
    #     FLAGS.model_architecture,
    #     is_training=True)
    #
    # # Define loss and optimizer
    # ground_truth_input = tf.placeholder(
    #     tf.float32, [None, label_count], name='groundtruth_input')
    #
    # # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # # numerical errors start occurring during training.
    # control_dependencies = []
    # if FLAGS.check_nans:
    #     checks = tf.add_check_numerics_ops()
    #     control_dependencies = [checks]
    #
    # # Create the back propagation and training evaluation machinery in the graph.
    # with tf.name_scope('cross_entropy'):
    #     cross_entropy_mean = tf.reduce_mean(
    #         tf.nn.softmax_cross_entropy_with_logits(
    #             labels=ground_truth_input, logits=logits))
    # tf.summary.scalar('cross_entropy', cross_entropy_mean)
    # with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    #     learning_rate_input = tf.placeholder(
    #         tf.float32, [], name='learning_rate_input')
    #     # train_step = tf.train.GradientDescentOptimizer(
    #     #     learning_rate_input).minimize(cross_entropy_mean)
    #     train_step = tf.train.AdamOptimizer(
    #         learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean)
    # predicted_indices = tf.argmax(logits, 1)
    # expected_indices = tf.argmax(ground_truth_input, 1)
    # correct_prediction = tf.equal(predicted_indices, expected_indices)
    # confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
    # evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', evaluation_step)
    #
    # global_step = tf.contrib.framework.get_or_create_global_step()
    # increment_global_step = tf.assign(global_step, global_step + 1)
    # saver = tf.train.Saver(tf.global_variables())
    # best_acc = 0.0
    # best_acc_saver = tf.train.Saver(tf.global_variables())
    # best_loss = 10.0
    # best_loss_saver = tf.train.Saver(tf.global_variables())
    # # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    # merged_summaries = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
    #                                      sess.graph)
    # validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    start_step = 1
    # if FLAGS.start_checkpoint:
    #     models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    #     start_step = global_step.eval(session=sess)
    tf.logging.info('Training from step: %d ', start_step)


    dcgan_model = DCGAN(
        batch_size=batch_size,
        image_shape=image_shape,
        dim_z=dim_z,
        dim_W1=dim_W1,
        dim_W2=dim_W2,
        dim_W3=dim_W3,
    )

    Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=10)

    discrim_vars = filter(lambda x: ('discrim' in x.name), tf.trainable_variables())
    gen_vars = filter(lambda x: ('gen' in x.name), tf.trainable_variables())
    discrim_vars = [i for i in discrim_vars]
    gen_vars = [i for i in gen_vars]

    train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
    train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

    # Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables())
    # Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim, dim_z))
    # Y_np_sample = OneHot(np.random.randint(10, size=[visualize_dim]))
    # iterations = 0
    k = 2

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
            FLAGS.background_volume, time_shift_samples, 'training', sess)
        Xs = train_fingerprints.reshape([-1,98,40,1])#trX[start:end].reshape([-1, 28, 28, 1]) / 255.
        Ys = train_ground_truth
        Zs = np.random.uniform(-1, 1, size=[train_ground_truth.shape[0], dim_z]).astype(np.float32)

        if np.mod(training_step, k) != 0:
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys
                })
            discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf, p_real, p_gen],
                                                               feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            if training_step % log_every == 0:
                # print("=========== updating G ==========")
                print("updating G\titeration:{}\t gen loss:{:.3}\t discrim loss:{:.3} ".format(
                      training_step, gen_loss_val, discrim_loss_val) )
                # print("iteration:", iterations)
                # print("gen loss:", gen_loss_val)
                # print("discrim loss:", discrim_loss_val)

        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    Z_tf: Zs,
                    Y_tf: Ys,
                    image_tf: Xs
                })
            gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen],
                                                           feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys})
            if training_step % log_every == 0:
                # print("=========== updating D ==========")
                print("updating D\titeration:{}\t gen loss:{:.3}\t discrim loss:{:.3} ".format(
                       training_step, gen_loss_val, discrim_loss_val))
                # print("gen loss:", gen_loss_val)
                # print("discrim loss:", discrim_loss_val)
        if training_step % log_every == 0:
            print("Average P(real)={:.3}\tAverage P(gen)={:.3}".format( p_real_val.mean(), p_gen_val.mean()) )
            # print("Average P(gen)=", p_gen_val.mean())
            saver.save(sess, os.path.join(FLAGS.train_dir, 'dcgan.ckpt'), global_step=training_step)
        # if np.mod(iterations, step) == 0:
        #     generated_samples = sess.run(
        #             image_tf_sample,
        #             feed_dict={
        #                 Z_tf_sample:Z_np_sample,
        #                 Y_tf_sample:Y_np_sample
        #                 })
        #     generated_samples = (generated_samples + 1.)/2.
        #     save_visualization(generated_samples, (14,14), save_path='./vis/sample_%04d.jpg' % int(iterations/step))



class ModelConfigureResnet(object):
    def __init__(self):
        self.summaries_dir='./logs_resnet'#retrain_logs
        self.lr_steps = '8000,4000,3000'  # 15000,10000,3000
        self.learning_rate = '0.001,0.0003,0.0001'
        self.batch_size = 64
        self.start_checkpoint = ''
        self.model_architecture='resnet'  # 'conv',
        self.train_dir ='./model_resnet'

if __name__ == '__main__':
    conf = ModelConfigureResnet()
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