import tensorflow as tf


def tf_split_reshape(line, part_n, delimiter=',', skip_empty=False):
    parts = tf.sparse_tensor_to_dense(tf.string_split(line, delimiter=delimiter, skip_empty=skip_empty),
                                      default_value='')
    parts = tf.reshape(parts, [part_n])
    return parts


def embedding_lookup(index, name, params):
    size = params[name + '_size']
    embed_size = params[name + '_embed_size']
    with tf.name_scope('{}_embedding'.format(name)):
        weights = tf.Variable(tf.truncated_normal([size, embed_size]), name='weights')
        embed = tf.nn.embedding_lookup(weights, index, name='lookup')
    return embed


def dense(data, layers, drop_rate=None, activation=tf.nn.relu, kernel_initializer=None, **kwargs):
    x = data
    for index, layer in enumerate(layers):
        x = tf.layers.dense(x, layer, activation=activation, kernel_initializer=kernel_initializer,
                            name='dense_layer{}'.format(index), **kwargs)
        if drop_rate:
            x = tf.layers.dropout(x, rate=drop_rate, name='dropout_layer{}'.format(index))
    return x


def cross_layers(data, degree):
    feature_n = data.get_shape().as_list()[1]
    xl = data
    for i in range(degree):
        weights = tf.Variable(tf.truncated_normal([feature_n, 1]), name='cross_layer{}_weights'.format(i))
        bias = tf.Variable(tf.truncated_normal([feature_n]), name='cross_layer{}_bias'.format(i))
        xl = tf.add(data * tf.matmul(xl, weights) + xl, bias, name='cross_layer{}'.format(i))
    return xl
