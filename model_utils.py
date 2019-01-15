import tensorflow as tf

def batch_norm_1(inputs, is_training ):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1,
        momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs

def merge_ops( x, person_embed, word_embed, is_training):
    # x = tf.layers.dense( x, units=128 )
    x = tf.layers.dense(x, 512)
    x = batch_norm_1( x, is_training )
    xm = x- person_embed
    xp = tf.concat( [xm, person_embed], axis=-1)
    xp = tf.layers.dense( xp, 512 )
    xp = batch_norm_1(xp, is_training)
    xd = word_embed+xp+x
    xd1 = word_embed+x
    xd2 = word_embed+xp
    xd = tf.concat([xd, xd1, xd2, xp, x], axis=-1)
    xw = tf.layers.dense( xd, 128 )
    xw = batch_norm_1( xw, is_training )
    return xw

# def merge_ops( x, person_embed, word_embed, is_training):
#     # x = tf.layers.dense( x, units=128 )
#     x = tf.layers.dense(x, 512)
#     x = batch_norm_1( x, is_training )
#     xm = x- person_embed
#     xp = tf.concat( [xm, person_embed], axis=-1)
#     xp = tf.layers.dense( xp, 512 )
#     xp = batch_norm_1(xp, is_training)
#     xd = word_embed+xp
#     xd = tf.concat([xd, xp], axis=-1)
#     xw = tf.layers.dense( xd, 128 )
#     xw = batch_norm_1( xw, is_training )
#     return xw


def get_layer_shape( layer):
    thisshape = tf.Tensor.get_shape(layer)
    ts = [thisshape[i].value for i in range(len(thisshape))]
    return ts