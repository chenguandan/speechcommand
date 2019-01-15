# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from resnet import create_resnet_model, create_resnet_model_keras



def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count):
    """Calculates common settings needed for all models.

    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_model(fingerprint_input, model_settings, model_architecture,
                 is_training, runtime_settings=None):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      model_architecture: String specifying which kind of model to create.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
      Exception: If the architecture type isn't recognized.
    """
    # if model_architecture == 'single_fc':
    #     return create_single_fc_model(fingerprint_input, model_settings,
    #                                   is_training)
    # elif model_architecture == 'conv':
    #     return create_conv_model(fingerprint_input, model_settings, is_training)
    # elif model_architecture == 'low_latency_conv':
    #     return create_low_latency_conv_model(fingerprint_input, model_settings,
    #                                          is_training)
    # elif model_architecture == 'low_latency_svdf':
    #     return create_low_latency_svdf_model(fingerprint_input, model_settings,
    #                                          is_training, runtime_settings)
    # elif model_architecture == 'resnet':
    #     return create_resnet_model(fingerprint_input, model_settings,
    #                                is_training)
    if model_architecture == 'cnnblstmp':
        return create_cnnblstm_model(fingerprint_input,model_settings,
                                     is_training)
    elif model_architecture == 'resnetblstmp':
        return create_resnetblstm_model(fingerprint_input,model_settings,
                                     is_training)
    # elif model_architecture == 'cnnblstm2':
    #     return create_cnnblstm_model_v2(fingerprint_input, model_settings,
    #                                  is_training)
    # elif model_architecture == 'cnnblstm6':
    #     return create_cnnblstm_model_v6(fingerprint_input, model_settings,
    #                                     is_training)
    # elif model_architecture == 'cnnblstmbnd':
    #     return create_cnnblstm_bnd_model_v6(fingerprint_input, model_settings,
    #                                     is_training)
    # elif model_architecture == 'lace':
    #     return create_lace(fingerprint_input, model_settings,
    #                                     is_training)
    # elif model_architecture == 'lace2':
    #     return create_lacev2(fingerprint_input, model_settings,
    #                                     is_training)

    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_single_fc_model(fingerprint_input, model_settings, is_training):
    """Builds a model with a single hidden fully-connected layer.

    This is a very simple model with just one matmul and bias layer. As you'd
    expect, it doesn't produce very accurate results, but it is very fast and
    simple, so it's useful for sanity testing.

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    fingerprint_size = model_settings['fingerprint_size']
    label_count = model_settings['label_count']
    weights = tf.Variable(
        tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
    bias = tf.Variable(tf.zeros([label_count]))
    logits = tf.matmul(fingerprint_input, weights) + bias
    if is_training:
        return logits, dropout_prob
    else:
        return logits


def create_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                              'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_conv_model(fingerprint_input, model_settings,
                                  is_training):
    """Builds a convolutional model with low compute requirements.

    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces slightly lower quality results than the 'conv' model, but needs
    fewer weight parameters and computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 4
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    first_conv_output_width = math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x)
    first_conv_output_height = math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y)
    first_conv_element_count = int(
        first_conv_output_width * first_conv_output_height * first_filter_count)
    flattened_first_conv = tf.reshape(first_dropout,
                                      [-1, first_conv_element_count])
    first_fc_output_channels = 128
    first_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_conv_element_count, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 128
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_low_latency_svdf_model(fingerprint_input, model_settings,
                                  is_training, runtime_settings):
    """Builds an SVDF model with low compute requirements.

    This is based in the topology presented in the 'Compressing Deep Neural
    Networks using a Rank-Constrained Topology' paper:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
          [SVDF]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This model produces lower recognition accuracy than the 'conv' model above,
    but requires fewer weight parameters and, significantly fewer computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      The node is expected to produce a 2D Tensor of shape:
        [batch, model_settings['dct_coefficient_count'] *
                model_settings['spectrogram_length']]
      with the features corresponding to the same time slot arranged contiguously,
      and the oldest slot at index [:, 0], and newest at [:, -1].
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
        ValueError: If the inputs tensor is incorrectly shaped.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    # Validation.
    input_shape = fingerprint_input.get_shape()
    if len(input_shape) != 2:
        raise ValueError('Inputs to `SVDF` should have rank == 2.')
    if input_shape[-1].value is None:
        raise ValueError('The last dimension of the inputs to `SVDF` '
                         'should be defined. Found `None`.')
    if input_shape[-1].value % input_frequency_size != 0:
        raise ValueError('Inputs feature dimension %d must be a multiple of '
                         'frame size %d', fingerprint_input.shape[-1].value,
                         input_frequency_size)

    # Set number of units (i.e. nodes) and rank.
    rank = 2
    num_units = 1280
    # Number of filters: pairs of feature and time filters.
    num_filters = rank * num_units
    # Create the runtime memory: [num_filters, batch, input_time_size]
    batch = 1
    memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                         trainable=False, name='runtime-memory')
    # Determine the number of new frames in the input, such that we only operate
    # on those. For training we do not use the memory, and thus use all frames
    # provided in the input.
    # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
    if is_training:
        num_new_frames = input_time_size
    else:
        window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                               model_settings['sample_rate'])
        num_new_frames = tf.cond(
            tf.equal(tf.count_nonzero(memory), 0),
            lambda: input_time_size,
            lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))
    new_fingerprint_input = fingerprint_input[
                            :, -num_new_frames * input_frequency_size:]
    # Expand to add input channels dimension.
    new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

    # Create the frequency filters.
    weights_frequency = tf.Variable(
        tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
    # Expand to add input channels dimensions.
    # weights_frequency: [input_frequency_size, 1, num_filters]
    weights_frequency = tf.expand_dims(weights_frequency, 1)
    # Convolve the 1D feature filters sliding over the time dimension.
    # activations_time: [batch, num_new_frames, num_filters]
    activations_time = tf.nn.conv1d(
        new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
    # Rearrange such that we can perform the batched matmul.
    # activations_time: [num_filters, batch, num_new_frames]
    activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

    # Runtime memory optimization.
    if not is_training:
        # We need to drop the activations corresponding to the oldest frames, and
        # then add those corresponding to the new frames.
        new_memory = memory[:, :, num_new_frames:]
        new_memory = tf.concat([new_memory, activations_time], 2)
        tf.assign(memory, new_memory)
        activations_time = new_memory

    # Create the time filters.
    weights_time = tf.Variable(
        tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
    # Apply the time filter on the outputs of the feature filters.
    # weights_time: [num_filters, input_time_size, 1]
    # outputs: [num_filters, batch, 1]
    weights_time = tf.expand_dims(weights_time, 2)
    outputs = tf.matmul(activations_time, weights_time)
    # Split num_units and rank into separate dimensions (the remaining
    # dimension is the input_shape[0] -i.e. batch size). This also squeezes
    # the last dimension, since it's not used.
    # [num_filters, batch, 1] => [num_units, rank, batch]
    outputs = tf.reshape(outputs, [num_units, rank, -1])
    # Sum the rank outputs per unit => [num_units, batch].
    units_output = tf.reduce_sum(outputs, axis=1)
    # Transpose to shape [batch, num_units]
    units_output = tf.transpose(units_output)

    # Appy bias.
    bias = tf.Variable(tf.zeros([num_units]))
    first_bias = tf.nn.bias_add(units_output, bias)

    # Relu.
    first_relu = tf.nn.relu(first_bias)

    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu

    first_fc_output_channels = 256
    first_fc_weights = tf.Variable(
        tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
    if is_training:
        second_fc_input = tf.nn.dropout(first_fc, dropout_prob)
    else:
        second_fc_input = first_fc
    second_fc_output_channels = 256
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    if is_training:
        final_fc_input = tf.nn.dropout(second_fc, dropout_prob)
    else:
        final_fc_input = second_fc
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def get_layer_shape( layer):
    thisshape = tf.Tensor.get_shape(layer)
    ts = [thisshape[i].value for i in range(len(thisshape))]
    return ts

def create_cnnblstm_model(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    from resnet import conv2d_fixed_padding, batch_norm_relu
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    person_count = model_settings['person_count']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    data_format = 'channels_last'
    conv1 = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=32, kernel_size=9, strides=1,
        data_format=data_format)
    conv1 = batch_norm_relu(conv1, is_training, data_format)
    print('layer1 shape',get_layer_shape(conv1))
    # inputs = tf.identity(inputs, 'initial_conv')
    conv2 = conv2d_fixed_padding(
        inputs=conv1, filters=64, kernel_size=4, strides=1,
        data_format=data_format)
    conv2 = batch_norm_relu(conv2, is_training, data_format)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=1, padding='SAME',
        data_format=data_format)
    pool2 = tf.identity(pool2, 'pool2')
    #TODO
    # conv_reshape = tf.squeeze(pool2, squeeze_dims=[1])
    pool2_shape = get_layer_shape(pool2)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(pool2,[-1,pool2_shape[1],pool2_shape[2]*pool2_shape[3]] )
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=128, kernel_size=1, strides=1,padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
    nb_hidden=128 #512
    seq_len=98

    # define for person
    # LSTM-1
    # with tf.variable_scope("cell_def_p1"):
    #     f_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
    #     b_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
    # with tf.variable_scope("cell_op_p1"):
    #     outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_rd,
    #                                                  sequence_length=None,
    #                                                  dtype=tf.float32)
    # mergep = tf.concat(outputs, 2)
    # shape = get_layer_shape(mergep)
    # print("First BLSTMp = ", shape)
    # LSTM-2
    # with tf.variable_scope("cell_def_p2"):
    #     f1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
    #     b1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
    #
    # with tf.variable_scope("cell_op_p2"):
    #     outputs2, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, mergep,
    #                                                   sequence_length=None,
    #                                                   dtype=tf.float32)
    # mergep2 = mergep
    # mergep2 = tf.concat(outputs2, 2)
    # convp1 = conv1d_fixed_padding(
    #     inputs=conv_rd, filters=128, kernel_size=4, strides=1,
    #     data_format=data_format)
    convp1 = tf.layers.conv1d(conv_rd, 128, kernel_size=4, strides=1,padding='same',
                              data_format=data_format)
    # convp1 = batch_norm_relu_1d(convp1, is_training, data_format)
    # convp2 = conv1d_fixed_padding(
    #     inputs=convp1, filters=256, kernel_size=4, strides=1,
    #     data_format=data_format)
    # convp2 = batch_norm_relu_1d(convp2, is_training, data_format)

    mergep2 = convp1
    # outp1 = tf.reduce_mean(mergep2, axis=1)
    # outp2 = tf.reduce_max(mergep2, axis=1)
    # outp = tf.concat([outp1, outp2], axis=-1)
    outp = tf.reduce_max(mergep2, axis=1)
    i_vec = tf.layers.dense(inputs=outp, units=100, name='i_vec')
    shape = get_layer_shape(outp)
    print("final feature p = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    outp = tf.layers.dense(inputs=outp, units=person_count, name='dense_fnp')
    # i_vecs = tf.rep
    shape = get_layer_shape(conv_rd)
    n_rep = shape[1]
    pattern = tf.stack([1, n_rep, 1])
    i_vec = tf.expand_dims(i_vec, 1)
    i_vecs = tf.tile( i_vec, pattern )
    shape = get_layer_shape(i_vecs)
    print('tile shape =>',shape)
    conv_rd = tf.concat( [conv_rd, i_vecs], axis=-1)
    #LSTM-1
    with tf.variable_scope("cell_def_1"):
        f_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
        b_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
    with tf.variable_scope("cell_op_1"):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_rd,
                                                     sequence_length=None,
                                                     dtype=tf.float32)

    merge = tf.concat( outputs, 2)
    shape = get_layer_shape(merge)
    print("First BLSTM = ", shape)
    #LSTM-2
    with tf.variable_scope("cell_def_2"):
        f1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
        b1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)

    with tf.variable_scope("cell_op_2"):
        outputs2, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, merge,
                                                      sequence_length=None,
                                                      dtype=tf.float32)

    merge2 = tf.concat(outputs2, 2)
    shape = get_layer_shape(merge2)
    print("Second BLSTM = ", shape)
    batch_s, timesteps = shape[0], shape[1]
    print(timesteps)
    blstm_features = shape[-1]

    # output_reshape = tf.reshape(merge2, [-1, blstm_features])  # maxsteps*batchsize,nb_hidden
    out1 = tf.reduce_mean(merge2, axis=1)
    out2 = tf.reduce_max(merge2, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')



    person_fc = tf.identity(outp, 'final_dense_smallp')

    if is_training:
        return final_fc, person_fc, dropout_prob
    else:
        return final_fc


from resnet import building_block, conv2d_fixed_padding,block_layer,batch_norm_relu
def create_resnetblstm_model(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    person_count = model_settings['person_count']
    print('shape',(input_time_size, input_frequency_size))
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    print((input_time_size, input_frequency_size))
    #setting
    data_format = 'channels_last'
    block_fn = building_block
    layers = [3, 4, 6, 3]

    inputs = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')
    #原本stride是1,2,2,2
    net1 = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    net2 = block_layer(
        inputs=net1, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    net3 = block_layer(
        inputs=net2, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    net4 = block_layer(
        inputs=net3, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)
    net4 = batch_norm_relu(net4, is_training, data_format)
    if is_training:
        net4 = tf.nn.dropout(net4, dropout_prob)

    #blstm
    pool2_shape = get_layer_shape(net4)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(net4, [-1, pool2_shape[1], pool2_shape[2] * pool2_shape[3]])
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1, padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)

    #person i-vector
    netp2 = block_layer(
        inputs=net1, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layerp2',
        data_format=data_format)
    netp3 = block_layer(
        inputs=netp2, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layerp3',
        data_format=data_format)
    netp4 = block_layer(
        inputs=netp3, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layerp4',
        data_format=data_format)
    conv_person = batch_norm_relu(netp4, is_training, data_format)
    shape = get_layer_shape(conv_person)
    # 原本strides=7
    inputs_person = tf.layers.average_pooling2d(
        inputs=conv_person, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
        data_format=data_format)
    inputs_person = tf.identity(inputs_person, 'final_avg_pool')
    outp = tf.reshape(inputs_person,
                         [-1, 512 if block_fn is building_block else 2048])
    i_vec = tf.layers.dense(inputs=outp, units=100, name='i_vec')
    shape = get_layer_shape(outp)
    print("final feature p = ", shape)
    outp = tf.layers.dense(inputs=outp, units=person_count, name='dense_fnp')
    #add i-vector to blstm
    shape = get_layer_shape(conv_rd)
    n_rep = shape[1]
    pattern = tf.stack([1, n_rep, 1])
    i_vec = tf.expand_dims(i_vec, 1)
    i_vecs = tf.tile(i_vec, pattern)
    shape = get_layer_shape(i_vecs)
    print('tile shape =>', shape)
    conv_rd = tf.concat([conv_rd, i_vecs], axis=-1)




    #fingerprint3d和这个shape不同
    rnn_input = conv_rd#tf.concat([conv_rd, fingerprint_3d], axis=-1)
    nb_hidden = 512
    seq_len = 98
    num_layer = 2
    rnn_output = rnn_input
    for layer_index in range(num_layer):
        with tf.variable_scope("cell_def_{}".format(layer_index)):
            f_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
            b_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)

        with tf.variable_scope("cell_op_{}".format(layer_index)):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, rnn_input,
                                                         sequence_length=None,
                                                         dtype=tf.float32)

        rnn_output = tf.concat(outputs, 2)
        shape = get_layer_shape(rnn_output)
        print(str(layer_index) + " BLSTM ", shape)
        rnn_input = rnn_output
    rnn_output = tf.concat([rnn_output, conv_rd], axis=-1)
    #TODO 没有使用:  使用：
    rnn_output = tf.layers.conv1d(rnn_output, 512,1,padding='valid')
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')
    person_fc = tf.identity(outp, 'final_dense_smallp')
    if is_training:
        return final_fc, person_fc, dropout_prob
    else:
        return final_fc



def create_cnnblstm_model_v2(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    from resnet import conv2d_fixed_padding, batch_norm_relu
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    data_format = 'channels_last'
    conv1 = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=256, kernel_size=9, strides=1,
        data_format=data_format)
    conv1 = batch_norm_relu(conv1, is_training, data_format)
    print('layer1 shape',get_layer_shape(conv1))
    # inputs = tf.identity(inputs, 'initial_conv')
    conv2 = conv2d_fixed_padding(
        inputs=conv1, filters=256, kernel_size=4, strides=1,
        data_format=data_format)
    conv2 = batch_norm_relu(conv2, is_training, data_format)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=1, padding='SAME',
        data_format=data_format)
    pool2 = tf.identity(pool2, 'pool2')
    #TODO
    # conv_reshape = tf.squeeze(pool2, squeeze_dims=[1])
    pool2_shape = get_layer_shape(pool2)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(pool2,[-1,pool2_shape[1],pool2_shape[2]*pool2_shape[3]] )
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1,padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
    rnn_input = tf.concat([conv_rd, fingerprint_3d], axis=-1)
    nb_hidden=512
    seq_len=98
    num_layer = 2
    rnn_output = rnn_input
    for layer_index in range(num_layer):
        with tf.variable_scope("cell_def_{}".format(layer_index)):
            f_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)
            b_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden, state_is_tuple=True)

        with tf.variable_scope("cell_op_{}".format(layer_index)):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, rnn_input,
                                                         sequence_length=None,
                                                         dtype=tf.float32)

        rnn_output = tf.concat( outputs, 2)
        shape = get_layer_shape(rnn_output)
        print(str(layer_index)+" BLSTM ", shape)
        rnn_input = rnn_output
    rnn_output = tf.concat([rnn_output, conv_rd], axis=-1)
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_cnnblstm_model_v6(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    from resnet import conv2d_fixed_padding, batch_norm_relu
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    data_format = 'channels_last'
    conv1 = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=64, kernel_size=9, strides=1,
        data_format=data_format)
    conv1 = batch_norm_relu(conv1, is_training, data_format)
    print('layer1 shape',get_layer_shape(conv1))
    # inputs = tf.identity(inputs, 'initial_conv')
    conv2 = conv2d_fixed_padding(
        inputs=conv1, filters=128, kernel_size=4, strides=1,
        data_format=data_format)
    conv2 = batch_norm_relu(conv2, is_training, data_format)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=1, padding='SAME',
        data_format=data_format)
    pool2 = tf.identity(pool2, 'pool2')
    #TODO
    # conv_reshape = tf.squeeze(pool2, squeeze_dims=[1])
    pool2_shape = get_layer_shape(pool2)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(pool2,[-1,pool2_shape[1],pool2_shape[2]*pool2_shape[3]] )
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1,padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
    rnn_input = tf.concat([conv_rd, fingerprint_3d], axis=-1)
    nb_hidden=512
    seq_len=98
    num_layer = 6
    rnn_output = rnn_input
    for layer_index in range(num_layer):
        with tf.variable_scope("cell_def_{}".format(layer_index)):
            f_cell = tf.nn.rnn_cell.GRUCell(nb_hidden)
            b_cell = tf.nn.rnn_cell.GRUCell(nb_hidden)

        with tf.variable_scope("cell_op_{}".format(layer_index)):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, rnn_input,
                                                         sequence_length=None,
                                                         dtype=tf.float32)

        rnn_output = tf.concat( outputs, 2)
        shape = get_layer_shape(rnn_output)
        print(str(layer_index)+" BLSTM ", shape)
        rnn_input = rnn_output
    rnn_output = tf.concat([rnn_output, conv_rd], axis=-1)
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_cnnblstm_bnd_model_v6(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    from resnet import conv2d_fixed_padding, batch_norm_relu
    import attention
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    data_format = 'channels_last'
    conv1 = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=64, kernel_size=9, strides=1,
        data_format=data_format)
    conv1 = batch_norm_relu(conv1, is_training, data_format)
    print('layer1 shape',get_layer_shape(conv1))
    # inputs = tf.identity(inputs, 'initial_conv')
    conv2 = conv2d_fixed_padding(
        inputs=conv1, filters=128, kernel_size=4, strides=1,
        data_format=data_format)
    conv2 = batch_norm_relu(conv2, is_training, data_format)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=3, strides=1, padding='SAME',
        data_format=data_format)
    pool2 = tf.identity(pool2, 'pool2')
    #TODO
    # conv_reshape = tf.squeeze(pool2, squeeze_dims=[1])
    pool2_shape = get_layer_shape(pool2)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(pool2,[-1,pool2_shape[1],pool2_shape[2]*pool2_shape[3]] )
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1,padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
    rnn_input = tf.concat([conv_rd, fingerprint_3d], axis=-1)
    nb_hidden=512
    seq_len=98
    num_layer = 4
    rnn_output = rnn_input
    for layer_index in range(num_layer):
        with tf.variable_scope("cell_def_{}".format(layer_index)):
            # f_cell = tf.nn.rnn_cell.GRUCell(nb_hidden)
            # b_cell = tf.nn.rnn_cell.GRUCell(nb_hidden)
            f_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(nb_hidden, dropout_keep_prob=0.9)
            b_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(nb_hidden, dropout_keep_prob=0.9)

        with tf.variable_scope("cell_op_{}".format(layer_index)):
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, rnn_input,
                                                         sequence_length=None,
                                                         dtype=tf.float32)

        rnn_output = tf.concat( outputs, 2)
        shape = get_layer_shape(rnn_output)
        print(str(layer_index)+" BLSTM ", shape)
        rnn_input = rnn_output
    rnn_output = tf.concat([rnn_output, conv_rd], axis=-1)
    rnn_output = conv1d_fixed_padding(
        inputs=rnn_output, filters=1024, kernel_size=4, strides=1,
        data_format=data_format)
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out3 = attention.attention(rnn_output, attention_size=128)
    out3 = tf.reshape(out3,[-1,1024] )
    out = tf.concat([out1, out2, out3], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc





def batch_norm_relu_1d(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 2,
        momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def conv1d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                             [0, 0]])
        inputs = padded_inputs
        # inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv1d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def block_layer_1d(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
    # filters_out = 4 * filters if block_fn is bottleneck_block else filters
    filters_out = filters
    def projection_shortcut(inputs):
        return conv1d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)

def building_block_1d(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
    """Standard building block for residual networks with BN before convolutions.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the convolutions.
      is_training: A Boolean for whether the model is in training or inference
        mode. Needed for batch normalization.
      projection_shortcut: The function to use for projection shortcuts (typically
        a 1x1 convolution when downsampling the input).
      strides: The block's stride. If greater than 1, this block will ultimately
        downsample the input.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_relu_1d(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv1d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu_1d(inputs, is_training, data_format)
    inputs = conv1d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut

def create_lace(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    data_format = 'channels_last'
    block_fn = building_block_1d
    layers = [3, 4, 6, 3]
    # conv1 = tf.layers.conv1d(
    #     inputs=fingerprint_3d, filters=64, kernel_size=3, strides=1,
    #     padding='SAME', use_bias=False,
    #     kernel_initializer=tf.variance_scaling_initializer(),
    #     data_format=data_format)
    # batch_norm_relu_1d(conv1, is_training, data_format)
    conv1 = conv1d_fixed_padding(
        inputs=fingerprint_3d, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.layers.max_pooling1d(
        inputs=conv1, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)

    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer_1d(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer_1d(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=1, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer_1d(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=1, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer_1d(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=1, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu_1d(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    # TODO pooling (3,7)
    inputs_ = tf.layers.average_pooling1d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    inputs_ = tf.identity(inputs_, 'final_avg_pool')
    fn_shape = get_layer_shape(inputs_)
    print('final shape => ', fn_shape)
    inputs_ = tf.reshape(inputs_,
                         [-1, fn_shape[1]*fn_shape[2] ])
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc_ = tf.identity(inputs_, 'final_dense')

    inputs = tf.layers.dense(inputs=final_fc_, units=label_count, name='dense_fn')
    final_fc = tf.identity(inputs, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def create_lacev2(fingerprint_input, model_settings,
                                  is_training):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    fingerprint_3d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size])
    data_format = 'channels_last'
    block_fn = building_block_1d
    layers = [5, 5, 5, 5]
    # conv1 = conv1d_fixed_padding(
    #     inputs=fingerprint_3d, filters=64, kernel_size=7, strides=2,
    #     data_format=data_format)
    # inputs = tf.layers.max_pooling1d(
    #     inputs=conv1, pool_size=3, strides=2, padding='SAME',
    #     data_format=data_format)

    inputs = tf.identity(fingerprint_3d, 'initial_max_pool')
    print('input  => ', get_layer_shape(inputs))
    inputs = block_layer_1d(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[0],
        strides=2, is_training=is_training, name='block_layer1',
        data_format=data_format)
    print('layer 1 => ',get_layer_shape(inputs))
    inputs = block_layer_1d(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    print('layer 2 => ', get_layer_shape(inputs))
    inputs = block_layer_1d(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer_1d(
        inputs=inputs, filters=1024, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)
    print('layer 3 => ', get_layer_shape(inputs))

    inputs = batch_norm_relu_1d(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    # TODO pooling (3,7)
    # inputs_ = tf.layers.average_pooling1d(
    #     inputs=inputs, pool_size=7, strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs_ = tf.identity(inputs_, 'final_avg_pool')
    inputs = conv1d_fixed_padding(inputs, 1024, kernel_size=4, strides=2,data_format=data_format)
    fn_shape = get_layer_shape(inputs)
    print('final shape => ', fn_shape)
    inputs_ = tf.reshape(inputs,
                         [-1, fn_shape[1]*fn_shape[2] ])
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc_ = tf.identity(inputs_, 'final_dense')

    inputs = tf.layers.dense(inputs=final_fc_, units=label_count, name='dense_fn')
    final_fc = tf.identity(inputs, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc