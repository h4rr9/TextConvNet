import tensorflow as tf
import os
import time
import utils
import numpy as np


train, val = utils.load_data()

train_data = tf.data.Dataset.from_tensor_slices(train)
# train_data = train_data.shuffle(10000)
train_data = train_data.batch(128)

val_data = tf.data.Dataset.from_tensor_slices(val)
val_data = val_data.batch(128)

iterator = tf.data.Iterator.from_structure(
    train_data.output_types, train_data.output_shapes
)

sentence, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
val_init = iterator.make_initializer(val_data)

with tf.name_scope("embed"):
    embedding_matrix = utils.load_embeddings()

    _embed = tf.constant(embedding_matrix)

    embed_matrix = tf.get_variable("embed_matrix", initializer=_embed)

    embed = tf.nn.embedding_lookup(embed_matrix, sentence, name="embed")

    embed = tf.reshape(embed, shape=[-1, 47, 300, 1])

writer = tf.summary.FileWriter(".\\graphs\\test-net", tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_init)
    try:
        while True:
            e = sess.run(embed)
    except tf.errors.OutOfRangeError:
        pass

writer.close()
