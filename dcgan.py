#-*- coding: utf-8 -*-
import tensorflow as tf
from models import get_layer_shape
from models import batch_norm_relu_1d
from resnet import batch_norm_relu

def print_layer_shape(Z, msg):
    shape = get_layer_shape(Z)
    print(msg, shape)

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=o, labels=t))

# class DCGAN():
#     def __init__(
#             self,
#             batch_size=100,
#             image_shape=[98,40,1],
#             dim_z=100,
#             dim_y=12,#包括silence/ unk
#             dim_W1=1024,
#             dim_W2=128,
#             dim_W3=64,
#             dim_channel=1,
#             ):
#
#         # self.batch_size = batch_size
#         self.image_shape = image_shape
#         self.dim_z = dim_z
#         self.dim_y = dim_y
#
#         self.dim_W1 = dim_W1
#         self.dim_W2 = dim_W2
#         self.dim_W3 = dim_W3
#         self.dim_channel = dim_channel
#
#         # self.gen_W1 = tf.Variable(tf.random_normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
#         # self.gen_W2 = tf.Variable(tf.random_normal([dim_W1+dim_y, dim_W2*7*7], stddev=0.02), name='gen_W2')
#         # self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
#         # self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')
#         #
#         # self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
#         # self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
#         # self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2*7*7+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
#         # self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')
#
#     def build_model(self):
#
#         Z = tf.placeholder(tf.float32, [None, self.dim_z])
#         Y = tf.placeholder(tf.float32, [None, self.dim_y])
#
#         image_real = tf.placeholder(tf.float32, [None]+self.image_shape)
#         h4 = self.generate(Z,Y)
#         image_gen = tf.nn.sigmoid(h4)
#         raw_real = self.discriminate(image_real, Y, reuse=False)
#         p_real = tf.nn.sigmoid(raw_real)
#         raw_gen = self.discriminate(image_gen, Y, reuse=True)
#         p_gen = tf.nn.sigmoid(raw_gen)
#         discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
#         discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
#         discrim_cost = discrim_cost_real + discrim_cost_gen
#
#         gen_cost = bce( raw_gen, tf.ones_like(raw_gen) )
#
#         return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen
#
#     def discriminate(self, image, Y, reuse):
#         with tf.variable_scope("dis", reuse=reuse):  # reuse the second time
#             yb = tf.reshape(Y, tf.stack([-1, 1, 1, self.dim_y]))
#             pattern = tf.stack([1, 98, 40, 1])
#             ybx = tf.tile(yb, pattern)
#             X = tf.concat(axis=3, values=[image, ybx])
#
#             # h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
#             h1 = tf.layers.conv2d(X, self.dim_W3, kernel_size=5, strides=2, padding='same', name='discrim_h1' )
#             h1 = tf.nn.leaky_relu(h1)
#             pattern = tf.stack([1, 49, 20, 1])
#             yb1 = tf.tile(yb, pattern)
#             h1 = tf.concat(axis=3, values=[h1, yb1])
#
#             # h2 = lrelu( batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
#             h2 = tf.layers.conv2d(h1, self.dim_W2, kernel_size=5, strides=(7,4), padding='same', name='discrim_h2')
#             h2 = tf.nn.leaky_relu(h2)
#             h2 = tf.reshape(h2, tf.stack([-1, self.dim_W2*7*5]))
#             h2 = tf.concat(axis=1, values=[h2, Y])
#
#             # h3 = lrelu( batchnormalize( tf.matmul(h2, self.discrim_W3 ) ))
#             h3 = tf.layers.dense(h2, self.dim_W1, use_bias=False, name='discrim_h3')
#             h3 = tf.nn.leaky_relu(h3)
#             h3 = tf.concat(axis=1, values=[h3, Y])
#         return h3
#
#     def generate(self, Z, Y):
#         is_training = True
#         data_format = 'channels_last'
#         yb = tf.reshape(Y, tf.stack([-1, 1, 1, self.dim_y]))
#         Z = tf.concat(axis=1, values=[Z,Y])
#         print_layer_shape(Z,'gen z ')
#         h1 = tf.layers.dense(Z, self.dim_W1, use_bias=False, name='gen_h1')
#         # h1 = batch_norm_relu_1d(h1, is_training, data_format)
#         h1 = tf.layers.batch_normalization(
#             inputs=h1, axis=1,
#             momentum=0.997, epsilon=1e-5, center=True,
#             scale=True, training=is_training, fused=True)
#         h1 = tf.nn.relu(h1)
#         print_layer_shape(h1, 'gen h1 ')
#         h1 = tf.concat(axis=1, values=[h1, Y])
#         h2 = tf.layers.dense(h1, self.dim_W2*7*5,use_bias=False, name='gen_h2')
#         h2 = tf.layers.batch_normalization(
#             inputs=h2, axis=1,
#             momentum=0.997, epsilon=1e-5, center=True,
#             scale=True, training=is_training, fused=True)
#         h2 = tf.nn.relu(h2)
#         # h2 = batch_norm_relu_1d(h2, is_training, data_format)
#         # h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
#         h2 = tf.reshape(h2, tf.stack([-1,7,5,self.dim_W2]))
#         pattern = tf.stack([1, 7, 5, 1])
#         yb2 = tf.tile(yb, pattern)
#         h2 = tf.concat(axis=3, values=[h2, yb2])
#         print_layer_shape(h2, 'gen h2 ')
#         # h2 = tf.concat(axis=3, values=[h2, yb*tf.ones([self.batch_size, 7, 7, self.dim_y])])
#
#         # output_shape_l3 = [-1,14,14,self.dim_W3]
#         # h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
#         # h3 = tf.nn.relu( batchnormalize(h3) )
#         # h3 = tf.concat(axis=3, values=[h3, yb*tf.ones([self.batch_size, 14,14,self.dim_y])] )
#         h3 = tf.layers.conv2d_transpose(h2, self.dim_W3, kernel_size=5, strides=(7,4),padding='same'
#                                         , name='gen_h3')
#         h3 = batch_norm_relu(h3, is_training, data_format)
#         pattern = tf.stack([1, 49, 20, 1])
#         yb3 = tf.tile(yb, pattern)
#         h3 = tf.concat(axis=3, values=[h3, yb3] )
#         print(h3, 'gen h3 ')
#         # output_shape_l4 = [self.batch_size,28,28,self.dim_channel]
#         # h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
#
#         h4 = tf.layers.conv2d_transpose(h3, kernel_size=5,filters=self.dim_channel,strides=(2,2),
#                                         padding='same',name='gen_h4' )
#         return h4
#
#     def samples_generator(self, batch_size):
#         Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
#         Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])
#
#         yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
#         Z_ = tf.concat(axis=1, values=[Z,Y])
#         h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
#         h1 = tf.concat(axis=1, values=[h1, Y])
#         h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
#         h2 = tf.reshape(h2, [batch_size,7,7,self.dim_W2])
#         h2 = tf.concat(axis=3, values=[h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])
#
#         output_shape_l3 = [batch_size,14,14,self.dim_W3]
#         h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
#         h3 = tf.nn.relu( batchnormalize(h3) )
#         h3 = tf.concat(axis=3, values=[h3, yb*tf.ones([batch_size, 14,14,self.dim_y])] )
#
#         output_shape_l4 = [batch_size,28,28,self.dim_channel]
#         h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
#         x = tf.nn.sigmoid(h4)
#         return Z,Y,x
#


from resnet import conv2d_fixed_padding

def batch_norm_lrelu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.leaky_relu(inputs)
    return inputs

def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   data_format):
    shortcut = inputs
    inputs = batch_norm_lrelu(inputs, is_training, data_format)
    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=5, strides=strides,
        data_format=data_format)

    inputs = batch_norm_lrelu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=5, strides=1,
        data_format=data_format)

    return inputs + shortcut

def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
                data_format):
    filters_out = filters

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


class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[98,40,1],
            dim_z=100,
            dim_y=12,#包括silence/ unk
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            ):

        # self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

    def build_model(self):

        Z = tf.placeholder(tf.float32, [None, self.dim_z])
        Y = tf.placeholder(tf.float32, [None, self.dim_y])

        image_real = tf.placeholder(tf.float32, [None]+self.image_shape)
        h4 = self.generate(Z,Y)
        image_gen = tf.nn.sigmoid(h4)
        raw_real = self.discriminate(image_real, Y, reuse=False)
        p_real = tf.nn.sigmoid(raw_real)
        raw_gen = self.discriminate(image_gen, Y, reuse=True)
        p_gen = tf.nn.sigmoid(raw_gen)
        discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
        discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
        discrim_cost = discrim_cost_real + discrim_cost_gen

        gen_cost = bce( raw_gen, tf.ones_like(raw_gen) )

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def concat_yb(self, h2_1, yb):
        shape = get_layer_shape(h2_1)
        pattern = tf.stack([1, shape[1], shape[2], 1])
        yb1 = tf.tile(yb, pattern)
        h2_1 = tf.concat(axis=3, values=[h2_1, yb1])
        return h2_1

    def discriminate(self, image, Y, reuse):
        is_training = True
        data_format = 'channels_last'
        with tf.variable_scope("dis", reuse=reuse):  # reuse the second time
            yb = tf.reshape(Y, tf.stack([-1, 1, 1, self.dim_y]))
            pattern = tf.stack([1, 98, 40, 1])
            ybx = tf.tile(yb, pattern)
            X = tf.concat(axis=3, values=[image, ybx])

            # h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
            h1 = tf.layers.conv2d(X, self.dim_W3, kernel_size=5, strides=2, padding='same', name='discrim_h1' )
            h1 = tf.nn.leaky_relu(h1)
            pattern = tf.stack([1, 49, 20, 1])
            yb1 = tf.tile(yb, pattern)
            h1 = tf.concat(axis=3, values=[h1, yb1])
            print_layer_shape(h1, 'dis h1 ')

            #resnet
            data_format = 'channels_last'
            block_fn = building_block
            layers = [1,1,1,1]#[3, 4, 6, 3]
            h2_1 = block_layer(
                inputs=h1, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=1, is_training=is_training, name='discrim_block_layer1',
                data_format=data_format)
            # shape = get_layer_shape(h2_1)
            # pattern = tf.stack([1, shape[1], shape[2], 1])
            # yb1 = tf.tile(yb, pattern)
            # h2_1 = tf.concat(axis=3, values=[h2_1, yb1])
            h2_1 = self.concat_yb(h2_1, yb)
            print_layer_shape(h2_1, 'dis h2_1 ')
            h2_2 = block_layer(
                inputs=h2_1, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='discrim_block_layer2',
                data_format=data_format)
            h2_2 = self.concat_yb(h2_2, yb)
            print_layer_shape(h2_2, 'dis h2_2 ')
            h2_3 = block_layer(
                inputs=h2_2, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=2, is_training=is_training, name='discrim_block_layer3',
                data_format=data_format)
            h2_3 = self.concat_yb(h2_3, yb)
            print_layer_shape(h2_3, 'dis h2_3 ')
            h2_4 = block_layer(
                inputs=h2_3, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=2, is_training=is_training, name='discrim_block_layer4',
                data_format=data_format)
            # h2_4 = self.concat_yb(h2_4, yb)

            # h2 = tf.layers.conv2d(h1, self.dim_W2, kernel_size=5, strides=(7,4), padding='same', name='discrim_h2')
            # h2 = tf.layers.batch_normalization(inputs=h2, axis=3,
            #                                    momentum=0.997, epsilon=1e-5, training=is_training, fused=True)
            # h2 = tf.nn.leaky_relu(h2)
            # h2 = tf.reshape(h2, tf.stack([-1, self.dim_W2*7*5]))
            # h2 = tf.concat(axis=1, values=[h2, Y])
            # print_layer_shape(h2, 'dis h2 ')
            # h2 = tf.nn.leaky_relu(h2)
            h2_4 = tf.layers.batch_normalization(h2_4, axis=3,
                                               momentum = 0.997, epsilon = 1e-5, training=is_training, fused=True)
            h2_4 = tf.nn.leaky_relu(h2_4)
            shape = get_layer_shape(h2_4)
            h2_4 = tf.reshape(h2_4, tf.stack([-1, shape[1]*shape[2]*shape[3]]))
            h2_4 = tf.concat([h2_4, Y], axis=1)
            print_layer_shape(h2_4, 'dis h2_4 ')
            h3 = tf.layers.dense(h2_4, self.dim_W1, use_bias=False, name='discrim_h3')
            h3 = tf.layers.batch_normalization(inputs=h3, axis=1,
                                               momentum = 0.997, epsilon = 1e-5, training=is_training, fused=True)
            h3 = tf.nn.leaky_relu(h3)
            h3 = tf.concat(axis=1, values=[h3, Y])
            print_layer_shape(h3, 'dis h3 ')
        return h3

    def generate(self, Z, Y):
        is_training = True
        data_format = 'channels_last'
        yb = tf.reshape(Y, tf.stack([-1, 1, 1, self.dim_y]))
        Z = tf.concat(axis=1, values=[Z,Y])
        print_layer_shape(Z,'gen z ')
        h1 = tf.layers.dense(Z, self.dim_W1, use_bias=False, name='gen_h1')
        # h1 = batch_norm_relu_1d(h1, is_training, data_format)
        h1 = tf.layers.batch_normalization(
            inputs=h1, axis=1,
            momentum=0.997, epsilon=1e-5, center=True,
            scale=True, training=is_training, fused=True)
        h1 = tf.nn.relu(h1)
        print_layer_shape(h1, 'gen h1 ')
        h1 = tf.concat(axis=1, values=[h1, Y])
        h2 = tf.layers.dense(h1, self.dim_W2*3*1, use_bias=False, name='gen_h2')
        h2 = tf.layers.batch_normalization(
            inputs=h2, axis=1,
            momentum=0.997, epsilon=1e-5, center=True,
            scale=True, training=is_training, fused=True)
        h2 = tf.nn.relu(h2)
        # h2 = batch_norm_relu_1d(h2, is_training, data_format)
        # h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, tf.stack([-1,3,1,self.dim_W2]))
        pattern = tf.stack([1, 3, 1, 1])
        yb2 = tf.tile(yb, pattern)
        h2 = tf.concat(axis=3, values=[h2, yb2])
        print_layer_shape(h2, 'gen h2 ')
        # h2 = tf.concat(axis=3, values=[h2, yb*tf.ones([self.batch_size, 7, 7, self.dim_y])])

        # output_shape_l3 = [-1,14,14,self.dim_W3]
        # h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        # h3 = tf.nn.relu( batchnormalize(h3) )
        # h3 = tf.concat(axis=3, values=[h3, yb*tf.ones([self.batch_size, 14,14,self.dim_y])] )
        #6,2
        h3 = tf.layers.conv2d_transpose(h2, self.dim_W3, kernel_size=5, strides=2, padding='same'
                                        , name='gen_h3')
        h3 = batch_norm_relu(h3, is_training, data_format)
        # pattern = tf.stack([1, 49, 20, 1])
        # yb3 = tf.tile(yb, pattern)
        # h3 = tf.concat(axis=3, values=[h3, yb3] )
        h3 = self.concat_yb(h3, yb)
        print_layer_shape(h3, 'gen h3 ')
        # output_shape_l4 = [self.batch_size,28,28,self.dim_channel]
        # h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        #12,4
        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=128, strides=2,
                                        padding='same', name='gen_h4_1')
        h4 = batch_norm_relu(h4, is_training, data_format)
        #12 5
        h4 = tf.pad(h4, [[0, 0], [0, 0],[0, 1], [0, 0]])
        h4 = self.concat_yb(h4, yb)
        print_layer_shape(h4, 'gen h4 1 ')
        #24 10
        h4 = tf.layers.conv2d_transpose(h4, kernel_size=5, filters=256, strides=2,
                                        padding='same', name='gen_h4_2')
        h4 = batch_norm_relu(h4, is_training, data_format)
        h4 = self.concat_yb(h4, yb)
        print_layer_shape(h4, 'gen h4 2 ')
        #48 20
        h4 = tf.layers.conv2d_transpose(h4, kernel_size=5, filters=512, strides=2,
                                        padding='same', name='gen_h4_3')
        h4 = batch_norm_relu(h4, is_training, data_format)
        #49 20
        h4 = tf.pad(h4, [[0, 0], [1, 0], [0, 0], [0, 0]])
        h4 = self.concat_yb(h4, yb)
        print_layer_shape(h4, 'gen h4 3 ')

        h4 = tf.layers.conv2d_transpose(h4, kernel_size=5,filters=self.dim_channel,strides=2,
                                        padding='same',name='gen_h4' )
        print_layer_shape(h4, 'gen h4')
        return h4

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat(axis=1, values=[Z,Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat(axis=1, values=[h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,7,7,self.dim_W2])
        h2 = tf.concat(axis=3, values=[h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [batch_size,14,14,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat(axis=3, values=[h3, yb*tf.ones([batch_size, 14,14,self.dim_y])] )

        output_shape_l4 = [batch_size,28,28,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z,Y,x


