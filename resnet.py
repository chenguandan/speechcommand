from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

from model_utils import get_layer_shape, merge_ops


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
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
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
    """Bottleneck block variant for residual networks with BN before convolutions.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first two convolutions. Note that the
        third and final convolution will use 4 times as many filters.
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
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
    """Creates one layer of blocks for the ResNet model.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      filters: The number of filters for the first convolution of the layer.
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      blocks: The number of blocks contained in the layer.
      strides: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
      is_training: Either True or False, whether we are currently training the
        model. Needed for batch norm.
      name: A string name for the tensor output of the block layer.
      data_format: The input format ('channels_last' or 'channels_first').
    Returns:
      The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)


def cifar10_resnet_v2_generator(resnet_size, num_classes, data_format=None):
    """Generator for CIFAR-10 ResNet v2 models.
    Args:
      resnet_size: A single integer for the size of the ResNet model.
      num_classes: The number of possible classes for image classification.
      data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
    Returns:
      The model function that takes in `inputs` and `is_training` and
      returns the output tensor of the ResNet model.
    Raises:
      ValueError: If `resnet_size` is invalid.
    """
    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=16, kernel_size=3, strides=1,
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = block_layer(
            inputs=inputs, filters=16, block_fn=building_block, blocks=num_blocks,
            strides=1, is_training=is_training, name='block_layer1',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=32, block_fn=building_block, blocks=num_blocks,
            strides=2, is_training=is_training, name='block_layer2',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=building_block, blocks=num_blocks,
            strides=2, is_training=is_training, name='block_layer3',
            data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=8, strides=1, padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs, [-1, 64])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def imagenet_resnet_v2_generator(block_fn, layers, num_classes,
                                 data_format=None):
    """Generator for ImageNet ResNet v2 models.
    Args:
      block_fn: The block to use within the model, either `building_block` or
        `bottleneck_block`.
      layers: A length-4 array denoting the number of blocks to include in each
        layer. Each layer consists of blocks that take inputs of the same size.
      num_classes: The number of possible classes for image classification.
      data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
    Returns:
      The model function that takes in `inputs` and `is_training` and
      returns the output tensor of the ResNet model.
    """
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """Constructs the ResNet model given the inputs."""
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=64, kernel_size=7, strides=2,
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')

        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1, is_training=is_training, name='block_layer1',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
            strides=2, is_training=is_training, name='block_layer2',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2, is_training=is_training, name='block_layer3',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, is_training=is_training, name='block_layer4',
            data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        inputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=7, strides=1, padding='VALID',
            data_format=data_format)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.reshape(inputs,
                            [-1, 512 if block_fn is building_block else 2048])
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        inputs = tf.identity(inputs, 'final_dense')
        return inputs

    return model


def imagenet_resnet_v2(resnet_size, num_classes, data_format=None):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {'block': building_block, 'layers': [2, 2, 2, 2]},
        34: {'block': building_block, 'layers': [3, 4, 6, 3]},
        50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)

    params = model_params[resnet_size]
    return imagenet_resnet_v2_generator(
        params['block'], params['layers'], num_classes, data_format)


def create_resnet_model(fingerprint_input, model_settings,
                                  is_training, use_fb=False):
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
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])
    print((input_time_size, input_frequency_size))
    # first_filter_width = 8
    # first_filter_height = input_time_size
    # first_filter_count = 384
    # first_filter_stride_x = 1
    # first_filter_stride_y = 1
    # first_weights = tf.Variable(
    #     tf.truncated_normal(
    #         [first_filter_height, first_filter_width, 1, first_filter_count],
    #         stddev=0.01))
    # first_bias = tf.Variable(tf.zeros([first_filter_count]))
    # first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
    #     1, first_filter_stride_y, first_filter_stride_x, 1
    # ], 'VALID') + first_bias
    # first_conv = tf.layers.conv2d(
    #     inputs=fingerprint_4d, filters=first_filter_count, kernel_size=(first_filter_height,
    #                                                                     first_filter_width), strides=1,
    #     padding='SAME', use_bias=True,
    #     kernel_initializer=tf.variance_scaling_initializer(),
    #     data_format="channels_last")
    # first_relu = tf.nn.relu(first_conv)
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
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #TODO pooling (3,7)
    from models import get_layer_shape
    shape = get_layer_shape(inputs)
    print('before final pool',shape)
    #原本strides=7
    inputs_ = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
        data_format=data_format)
    inputs_ = tf.identity(inputs_, 'final_avg_pool')
    inputs_ = tf.reshape(inputs_,
                        [-1, 512 if block_fn is building_block else 2048])
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc = tf.identity(inputs_, 'final_dense')

    # inputs = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=(3,7), strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs1 = tf.reduce_mean(inputs, axis=[1,2])
    # inputs2 = tf.reduce_max(inputs,axis=[1,2])
    # inputs = tf.concat([inputs1, inputs2], axis=-1)
    # inputs = tf.identity(inputs, 'final_avg_pool_small')
    # inputs = tf.reshape(inputs,
    #                      [-1, 512*2 if block_fn is building_block else 2048*2])
    # inputs = tf.layers.dense(inputs=inputs, units=label_count, name='dense_fn')
    # final_fc = tf.identity(inputs, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def create_resnet_model_p(fingerprint_input, person_embed, word_embed, model_settings,
                                  is_training, use_fb=False):
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
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
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
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #TODO pooling (3,7)
    from models import get_layer_shape
    shape = get_layer_shape(inputs)
    print('before final pool',shape)
    #原本strides=7
    inputs_ = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
        data_format=data_format)
    inputs_ = tf.identity(inputs_, 'final_avg_pool')
    inputs_ = tf.reshape(inputs_,
                        [-1, 512 if block_fn is building_block else 2048])
    inputs_ = merge_ops( inputs_, person_embed, word_embed, is_training)
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc = tf.identity(inputs_, 'final_dense')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def resnet_embed(fingerprint_4d, is_training, dropout_prob, reuse):
    data_format = 'channels_last'
    block_fn = building_block
    layers = [3, 4, 6, 3]
    with tf.variable_scope("resnet", reuse=reuse):
        inputs = conv2d_fixed_padding(
            inputs=fingerprint_4d, filters=64, kernel_size=7, strides=2,
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_conv')
        inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=data_format)
        inputs = tf.identity(inputs, 'initial_max_pool')
        # 原本stride是1,2,2,2
        inputs = block_layer(
            inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
            strides=1, is_training=is_training, name='block_layer1',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
            strides=2, is_training=is_training, name='block_layer2',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
            strides=2, is_training=is_training, name='block_layer3',
            data_format=data_format)
        inputs = block_layer(
            inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
            strides=2, is_training=is_training, name='block_layer4',
            data_format=data_format)

        inputs = batch_norm_relu(inputs, is_training, data_format)
        if is_training:
            inputs = tf.nn.dropout(inputs, dropout_prob)

        from models import get_layer_shape
        shape = get_layer_shape(inputs)
        print('before final pool', shape)
        # 原本strides=7
        inputs_ = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
            data_format=data_format)
        inputs_ = tf.identity(inputs_, 'final_avg_pool')
        inputs_ = tf.reshape(inputs_,
                             [-1, 512 if block_fn is building_block else 2048])
        inputs_ = tf.layers.dense(inputs=inputs_, units=128)
        final_fc = tf.identity(inputs_, 'final_dense')
    return final_fc


def create_resnet_pairs_model(fingerprint_input1, fingerprint_input2, model_settings,
                                  is_training, use_fb=False):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d1 = tf.reshape(fingerprint_input1,
                                    [-1, input_time_size+1, 26, 1])
        fingerprint_4d2 = tf.reshape(fingerprint_input2,
                                    [-1, input_time_size + 1, 26, 1])
    else:
        fingerprint_4d1 = tf.reshape(fingerprint_input1,
                                    [-1, input_time_size, input_frequency_size, 1])
        fingerprint_4d2 = tf.reshape(fingerprint_input2,
                                    [-1, input_time_size, input_frequency_size, 1])
    print((input_time_size, input_frequency_size))
    #setting
    embed1 = resnet_embed(fingerprint_4d1, is_training, dropout_prob, reuse=False)
    embed2 = resnet_embed(fingerprint_4d2, is_training, dropout_prob, reuse=True)
    ds = tf.layers.dense( tf.concat( [embed1*embed2, embed1+embed2, tf.abs(embed1-embed2)], axis=-1), 128 )
    ds = tf.nn.leaky_relu(ds)
    logits = tf.layers.dense( ds, 2 )



    if is_training:
        return embed1, embed2, logits, dropout_prob
    else:
        return embed1, embed2, logits



def create_resnetblstm_model(fingerprint_input, model_settings,
                                  is_training, use_fb=False):
    """
    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    from models import get_layer_shape
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
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
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    # inputs = block_layer(
    #     inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
    #     strides=2, is_training=is_training, name='block_layer4',
    #     data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #blstm
    pool2_shape = get_layer_shape(inputs)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(inputs, [-1, pool2_shape[1], pool2_shape[2] * pool2_shape[3]])
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1, padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
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
    # rnn_output = tf.layers.conv1d(rnn_output, 512,1,padding='valid')
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')

    #TODO pooling (3,7)
    # shape = get_layer_shape(inputs)
    # print('before final pool',shape)
    # #原本strides=7
    # inputs_ = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs_ = tf.identity(inputs_, 'final_avg_pool')
    # inputs_ = tf.reshape(inputs_,
    #                     [-1, 512 if block_fn is building_block else 2048])
    # inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    # final_fc = tf.identity(inputs_, 'final_dense')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


def create_resnetblstm_model_p(fingerprint_input, person_embed, word_embed,model_settings,
                                  is_training, use_fb=False):
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
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
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
    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    # inputs = block_layer(
    #     inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
    #     strides=2, is_training=is_training, name='block_layer4',
    #     data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #blstm
    pool2_shape = get_layer_shape(inputs)
    print('layer2 shape', pool2_shape)
    conv_reshape = tf.reshape(inputs, [-1, pool2_shape[1], pool2_shape[2] * pool2_shape[3]])
    shape = get_layer_shape(conv_reshape)
    print("CNN --> RNN Reshape = ", shape)
    conv_rd = tf.layers.conv1d(
        inputs=conv_reshape, filters=256, kernel_size=1, strides=1, padding='same',
        data_format=data_format)
    shape = get_layer_shape(conv_rd)
    print("conv reduce dim = ", shape)
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
    # rnn_output = tf.layers.conv1d(rnn_output, 512,1,padding='valid')
    out1 = tf.reduce_mean(rnn_output, axis=1)
    out2 = tf.reduce_max(rnn_output, axis=1)
    out = tf.concat([out1, out2], axis=-1)
    shape = get_layer_shape(out)
    print("final feature = ", shape)
    # if is_training:
    #     out = tf.nn.dropout(out, dropout_prob)
    out = merge_ops(out, person_embed, word_embed, is_training)
    out = tf.layers.dense(inputs=out, units=label_count, name='dense_fn')
    final_fc = tf.identity(out, 'final_dense_small')

    #TODO pooling (3,7)
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def create_resnet_model101(fingerprint_input, model_settings,
                                  is_training, use_fb=False):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])
    print((input_time_size, input_frequency_size))
    #setting
    data_format = 'channels_last'
    block_fn = bottleneck_block
    layers = [3, 4, 23, 3]
    # block_fn = building_block
    # layers = [3, 4, 6, 3]

    inputs = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #TODO pooling (3,7)
    from models import get_layer_shape
    shape = get_layer_shape(inputs)
    print('before final pool', shape)
    # 原本strides=7
    inputs_ = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
        data_format=data_format)
    inputs_ = tf.identity(inputs_, 'final_avg_pool')
    inputs_ = tf.reshape(inputs_,
                        [-1, 512 if block_fn is building_block else 2048])
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc = tf.identity(inputs_, 'final_dense')

    # inputs = tf.layers.average_pooling2d(
    #     inputs=inputs, pool_size=(3,7), strides=1, padding='VALID',
    #     data_format=data_format)
    # inputs1 = tf.reduce_mean(inputs, axis=[1,2])
    # inputs2 = tf.reduce_max(inputs,axis=[1,2])
    # inputs = tf.concat([inputs1, inputs2], axis=-1)
    # inputs = tf.identity(inputs, 'final_avg_pool_small')
    # inputs = tf.reshape(inputs,
    #                      [-1, 512*2 if block_fn is building_block else 2048*2])
    # inputs = tf.layers.dense(inputs=inputs, units=label_count, name='dense_fn')
    # final_fc = tf.identity(inputs, 'final_dense_small')
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc

def create_resnet_model101_p(fingerprint_input, person_embed, word_embed,model_settings,
                                  is_training, use_fb=False):
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    label_count = model_settings['label_count']
    print('shape',(input_time_size, input_frequency_size))
    if use_fb:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size+1, 26, 1])
    else:
        fingerprint_4d = tf.reshape(fingerprint_input,
                                    [-1, input_time_size, input_frequency_size, 1])
    print((input_time_size, input_frequency_size))
    #setting
    data_format = 'channels_last'
    block_fn = bottleneck_block
    layers = [3, 4, 23, 3]
    # block_fn = building_block
    # layers = [3, 4, 6, 3]

    inputs = conv2d_fixed_padding(
        inputs=fingerprint_4d, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    inputs = block_layer(
        inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)
    inputs = block_layer(
        inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
        strides=2, is_training=is_training, name='block_layer4',
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    if is_training:
        inputs = tf.nn.dropout(inputs, dropout_prob)

    #TODO pooling (3,7)
    from models import get_layer_shape
    shape = get_layer_shape(inputs)
    print('before final pool', shape)
    # 原本strides=7
    inputs_ = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=(shape[1], shape[2]), strides=1, padding='VALID',
        data_format=data_format)
    inputs_ = tf.identity(inputs_, 'final_avg_pool')
    inputs_ = tf.reshape(inputs_,
                        [-1, 512 if block_fn is building_block else 2048])
    inputs_ = merge_ops(inputs_, person_embed, word_embed, is_training)
    inputs_ = tf.layers.dense(inputs=inputs_, units=label_count)
    final_fc = tf.identity(inputs_, 'final_dense')

    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_resnet_model_keras(fingerprint_input, model_settings,
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
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 384
    first_filter_stride_x = 1
    first_filter_stride_y = 1
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    # first_conv = tf.layers.conv2d(
    #     inputs=fingerprint_4d, filters=first_filter_count, kernel_size=(first_filter_height,
    #                                                                     first_filter_width), strides=1,
    #     padding='SAME', use_bias=True,
    #     kernel_initializer=tf.variance_scaling_initializer(),
    #     data_format="channels_last")
    first_relu = tf.nn.relu(first_conv)
    #setting
    data_format = 'channels_last'

    inputs = conv2d_fixed_padding(
        inputs=first_relu, filters=64, kernel_size=7, strides=2,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME',
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_max_pool')

    img_input = Input(tensor=inputs)#, shape=input_shape
    bn_axis = 3

    x = ZeroPadding2D((3, 3))(img_input)
    # x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    # x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # x = AveragePooling2D((2, 7), name='avg_pool')(x)#7,7

    x = Dense(label_count, activation='softmax', name='fc1000')(x)


    model = Model(img_input, x, name='resnet50')
    model.summary( )
    # return model
    if is_training:
        return x, dropout_prob
    else:
        return x
