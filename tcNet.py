import tensorflow as tf
import numpy as np
import time
import utils
import os
import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PATH_CHECKPOINTS = './checkpoints'
# PATH_CHECKPOINTS = '/scratch/scratch5/harig/tcNet_sub/checkpoints'
PATH_GRAPHS = './graphs'
# PATH_GRAPHS = '/scratch/scratch5/harig/tcNet_sub/graphs'

PATH_BEST_WEIGHTS = './best_weights'
# PATH_BEST_WEIGHTS = '../Trained Weights/tc-sub-data/checkpoints'

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4


class TextConvNet:

    def __init__(self):
        self.batch_size = 50
        self.learning_rate = 1
        self.l2_constraint = 3
        self.n_classes = 2
        self.skip_step = 20
        self.training = False
        self.best_acc = 0

    def init(self):
        self.train_prob = 0.5
        self.test_prob = 1.0
        self.keep_prob = tf.placeholder(
            name='keep_prob', shape=[], dtype=tf.float32)

        self.gstep = tf.get_variable('global_step',
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.int32, trainable=False, shape=[])
        self.vstep = tf.get_variable('validation_step',
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.int32, trainable=False, shape=[])

    def import_data(self):
        train, val = utils.load_sst2_data()

        self.max_sentence_size = train[0].shape[1]
        self.n_train = train[0].shape[0]
        self.n_test = val[0].shape[0]

        train_data = tf.data.Dataset.from_tensor_slices(train)
        train_data = train_data.shuffle(self.n_train)
        train_data = train_data.batch(self.batch_size)

        val_data = tf.data.Dataset.from_tensor_slices(val)
        val_data = val_data.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(
            train_data.output_types, train_data.output_shapes)

        self.sentence, self.label = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        self.val_init = iterator.make_initializer(val_data)

    def get_embedding(self, _weights=None):
        with tf.name_scope('embed'):

            if _weights is None:
                embedding_matrix = utils.load_embeddings_sst2()
                _embed = tf.constant(embedding_matrix)
                embed_matrix = tf.get_variable(
                    'embed_matrix', initializer=_embed)
            else:
                _embed = tf.constant(_weights)
                embed_matrix = tf.get_variable(
                    'embed_matrix', initializer=_embed)

            self.embed = tf.nn.embedding_lookup(
                embed_matrix, self.sentence, name='embed')

    def model(self):

        conv0 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=3,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv0')
        pool0 = layers.one_maxpool(inputs=conv0,
                                   padding='VALID', scope_name='pool0')

        flatten0 = layers.flatten(pool0, scope_name='flatten0')

        conv1 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=4,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv1')
        pool1 = layers.one_maxpool(inputs=conv1,
                                   padding='VALID', scope_name='pool1')

        flatten1 = layers.flatten(pool1, scope_name='flatten1')

        conv2 = layers.conv1d_relu(inputs=self.embed,
                                   filters=100,
                                   k_size=5,
                                   stride=1,
                                   padding='SAME',
                                   scope_name='conv2')
        pool2 = layers.one_maxpool(inputs=conv2,
                                   padding='VALID', scope_name='pool2')

        flatten2 = layers.flatten(inputs=pool2, scope_name='flatten2')

        concat0 = layers.concatinate(
            inputs=[flatten0, flatten1, flatten2], scope_name='concat0')

        dropout0 = layers.Dropout(
            inputs=concat0, rate=1 - self.keep_prob, scope_name='dropout0')

        self.logits = layers.fully_connected(
            inputs=dropout0, out_dim=self.n_classes, scope_name='fc0')

    def init_loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.label, logits=self.logits)
            loss = tf.reduce_mean(entropy, name='loss')

            vars = [v for v in tf.trainable_variables() if 'fc' in v.name]

            l2_norm = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            self.loss = loss + self.l2_constraint * l2_norm

    def init_optimize(self):
        with tf.name_scope('optimize'):
            _opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate)
            self.opt = _opt.minimize(self.loss, global_step=self.gstep)

    def init_summaries(self):
        with tf.name_scope('train_summaries'):
            train_loss = tf.summary.scalar('train_loss', self.loss)
            train_accuracy = tf.summary.scalar(
                'train_accuracy', self.accuracy / self.batch_size)
            hist_train_loss = tf.summary.histogram(
                'histogram_train_loss', self.loss)
            self.train_summary_op = tf.summary.merge(
                [train_loss, train_accuracy, hist_train_loss])

        with tf.name_scope('val_summaries'):
            val_loss = tf.summary.scalar('val_loss', self.loss)
            val_summary = tf.summary.scalar(
                'val_accuracy', self.accuracy / self.batch_size)
            hist_val_loss = tf.summary.histogram(
                'histogram_val_loss', self.loss)
            self.val_summary_op = tf.summary.merge(
                [val_loss, val_summary, hist_val_loss])

    def init_eval(self):
        with tf.name_scope('eval'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

            self.increment_vstep = tf.assign_add(
                self.vstep, 1, name='increment_vstep')

    def build(self):

        self.init()
        self.import_data()
        self.get_embedding()
        self.model()
        self.init_loss()
        self.init_optimize()
        self.init_eval()
        self.init_summaries()

    def train_one_epoch(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        total_correct_preds = 0

        try:
            while True:
                _, l, accuracy_batch, summaries = sess.run(
                    [self.opt,
                     self.loss,
                     self.accuracy,
                     self.train_summary_op], feed_dict={self.keep_prob: self.train_prob})
                writer.add_summary(summaries, global_step=step)

                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step = step + 1
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
        except tf.errors.OutOfRangeError:
            pass

        print('\nAverage training loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Training accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_train))
        print('Took: {0} seconds'.format(time.time() - start_time))

        return step

    def eval_once(self, sess, best_saver, saver, init, writer, epoch, val_step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, accuracy_batch, summaries = sess.run(
                    [self.increment_vstep, self.loss, self.accuracy, self.val_summary_op], feed_dict={self.keep_prob: self.test_prob})
                writer.add_summary(summaries, global_step=val_step)
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
                val_step = val_step + 1
        except tf.errors.OutOfRangeError:
            pass

        if self.best_acc < total_correct_preds / self.n_test:
            print('\nSaving best accuracy: {0} from {1}\n'.format(
                total_correct_preds / self.n_test, self.best_acc))
            self.best_acc = total_correct_preds / self.n_test
            best_saver.save(sess, PATH_BEST_WEIGHTS + '/model', self.gstep)
        else:
            print('\nBest Accuracy unchanged from : {0}\n'.format(
                self.best_acc))
        saver.save(sess, PATH_CHECKPOINTS + '/model', self.gstep)

        print('Average validation loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Validation accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds\n'.format(time.time() - start_time))

        return val_step

    def train(self, n_epochs, resume=False):
        utils.mkdir_safe(os.path.dirname(PATH_CHECKPOINTS))
        utils.mkdir_safe(PATH_CHECKPOINTS)
        utils.mkdir_safe(PATH_BEST_WEIGHTS)

        train_writer = tf.summary.FileWriter(
            PATH_GRAPHS + '/train', tf.get_default_graph())

        val_writer = tf.summary.FileWriter(
            PATH_GRAPHS + '/val')

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            best_saver = tf.train.Saver(max_to_keep=1)

            if resume == True:
                ckpt = tf.train.get_checkpoint_state(PATH_CHECKPOINTS)

                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()
            val_step = self.vstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(
                    sess, self.train_init, train_writer, epoch, step)
                val_step = self.eval_once(
                    sess, best_saver, saver, self.val_init, val_writer, epoch, val_step)

        train_writer.close()
        val_writer.close()


if __name__ == '__main__':

    model = TextConvNet()
    model.build()
    model.train(n_epochs=100)
