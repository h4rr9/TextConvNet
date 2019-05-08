import tensorflow as tf


def conv1d_relu(inputs, filters, k_size, stride, padding, scope_name='conv', _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        in_channels = inputs.shape[-1]

        if _weights is None:
            kernel = tf.get_variable('kernel',
                                     [k_size, in_channels, filters],
                                     initializer=tf.truncated_normal_initializer())

            biases = tf.get_variable('biases',
                                     [filters],
                                     initializer=tf.random_normal_initializer())
        else:
            kernel = tf.get_variable(
                'kernel', initializer=tf.constant(_weights[0]))

            biases = tf.get_variable(
                'biases', initializer=tf.constant(_weights[1]))
        conv = tf.nn.conv1d(inputs,
                            kernel,
                            stride=stride,
                            padding=padding)

        output = tf.nn.relu(conv + biases, name=scope.name)

    return output


def one_maxpool(inputs, padding='VALID', scope_name='one-pool1d'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        height, in_channel = inputs.shape[-2:]

        pool = tf.nn.pool(input=inputs, window_shape=[
                          height], pooling_type='MAX', padding=padding, strides=[1], name=scope.name)

    return pool


def maxpool1d(inputs, k_size, stride=None, padding='VALID', scope_name='pool1d'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        if stride is None:
            stride = k_size

        pool = tf.nn.pool(input=inputs, window_shape=[
                          k_size], pooling_type='MAX', padding=padding, strides=[stride], name=scope.name)

        return pool


def flatten(inputs, scope_name='flatten'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        feature_dim = inputs.shape[1] * inputs.shape[2]

        flatten = tf.reshape(inputs, shape=[-1, feature_dim], name=scope.name)

    return flatten


def concatinate(inputs, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        concat = tf.concat(inputs, 1, name=scope.name)

    return concat


def fully_connected(inputs, out_dim, scope_name='fc', _weights=None):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:

        if _weights is None:
            in_dim = inputs.shape[-1]
            w = tf.get_variable('weights',
                                [in_dim, out_dim],
                                initializer=tf.truncated_normal_initializer())

            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer(0.0))
        else:
            w = tf.get_variable(
                'weights', initializer=tf.constant(_weights[0]))

            biases = tf.get_variable(
                'biases', initializer=tf.constant(_weights[1]))


        out = tf.add(tf.matmul(inputs, w), b, name=scope.name)
    return out


def Dropout(inputs, rate, scope_name='dropout'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        dropout = tf.nn.dropout(inputs, keep_prob=1 - rate, name=scope.name)
    return dropout


def l2_norm(inputs, alpha, scope_name='l2_norm'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        norm = alpha * tf.divide(inputs,
                                 tf.norm(inputs, ord='euclidean'),
                                 name=scope.name)
    return norm
