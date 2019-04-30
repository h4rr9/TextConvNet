import tensorflow as tf
import time
import utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def conv_relu(inputs, filters, k_size, stride, padding, scope_name='conv'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = 1
        width = inputs.shape[2]

        kernel = tf.get_variable('kernal',
                                 [k_size, width, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())
        biases = tf.get_variable('biases',
                                 [filters],
                                 initializer=tf.random_normal_initializer())

        conv = tf.nn.conv2d(inputs,
                            kernel,
                            strides=[1, stride, stride, 1],
                            padding=padding)

    return tf.nn.relu(conv + biases, name=scope.name)


def maxpool(inputs, stride, padding='VALID', scope_name='pool'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        height = inputs.shape[1]
        width = inputs.shape[2]
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, height, width, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding,
                              name=scope.name)

    return pool


def flatten(inputs, scope_name='flatten'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        feature_dim = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]

        flatten = tf.reshape(inputs, shape=[-1, feature_dim], name=scope.name)

    return flatten


def concatinate(inputs, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        concat = tf.concat(inputs, 1, name=scope.name)

    return concat


def fully_connected(inputs, out_dim, scope_name='fc'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights',
                            [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer())

        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(0.0))

        out = tf.add(tf.matmul(inputs, w), b, name=scope.name)
    return out


def Dropout(inputs, rate, scope_name='dropout'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        dropout = tf.nn.dropout(inputs, rate=rate, name=scope.name)
    return dropout


def l2_norm(inputs, alpha, scope_name='l2_norm'):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        norm = alpha * tf.divide(inputs,
                                 tf.norm(inputs, ord='euclidean'),
                                 name=scope.name)
    return norm


class TextConvNet:

    def __init__(self):
        self.batch_size = 50
        self.learning_rate = 1.0
        self.l2_contraint = 3
        self.gstep = tf.get_variable('global_step',
                                     initializer=tf.constant_initializer(0),
                                     dtype=tf.int32,
                                     trainable=False,
                                     shape=[])
        self.n_classes = 2
        self.skip_step = 20
        self.training = False
        self.keep_prob = tf.constant(0.5)

    def import_data(self):
        train, val = utils.load_subjectivity_data()

        self.max_sentence_size = train[0].shape[1]
        self.n_train = train[0].shape[0]
        self.n_test = val[0].shape[0]

        train_data = tf.data.Dataset.from_tensor_slices(train)
        train_data = train_data.shuffle(9595)
        train_data = train_data.batch(self.batch_size)

        val_data = tf.data.Dataset.from_tensor_slices(val)
        val_data = val_data.batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(
            train_data.output_types, train_data.output_shapes)

        self.sentence, self.label = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        self.val_init = iterator.make_initializer(val_data)

    def get_embedding(self):
        with tf.name_scope('embed'):
            embedding_matrix = utils.load_embeddings_subjectivity()
            _embed = tf.constant(embedding_matrix)
            embed_matrix = tf.get_variable(
                'embed_matrix', initializer=_embed)

            embed = tf.nn.embedding_lookup(
                embed_matrix, self.sentence, name='embed')

            self.embed = tf.reshape(
                embed, shape=[-1, self.max_sentence_size, 300, 1])

    def model(self):
        conv0 = conv_relu(inputs=self.embed,
                          filters=100,
                          k_size=3,
                          stride=1,
                          padding='VALID',
                          scope_name='conv0')
        pool0 = maxpool(inputs=conv0, stride=1,
                        padding='VALID', scope_name='pool0')
        flatten0 = flatten(inputs=pool0, scope_name='flatten0')

        conv1 = conv_relu(inputs=self.embed,
                          filters=100,
                          k_size=4,
                          stride=1,
                          padding='VALID',
                          scope_name='conv1')
        pool1 = maxpool(inputs=conv1, stride=1,
                        padding='VALID', scope_name='pool1')
        flatten1 = flatten(inputs=pool1, scope_name='flatten1')

        conv2 = conv_relu(inputs=self.embed,
                          filters=100,
                          k_size=5,
                          stride=1,
                          padding='VALID',
                          scope_name='conv2')
        pool2 = maxpool(inputs=conv2, stride=1,
                        padding='VALID', scope_name='pool2')
        flatten2 = flatten(inputs=pool2, scope_name='flatten2')

        concat0 = concatinate(
            inputs=[flatten0, flatten1, flatten2], scope_name='concat0')

        norm0 = l2_norm(concat0, alpha=self.l2_contraint, scope_name='norm0')

        dropout0 = Dropout(
            inputs=norm0, rate=1 - self.keep_prob, scope_name='dropout0')

        self.logits_train = fully_connected(
            inputs=dropout0, out_dim=self.n_classes, scope_name='fc0')

        self.logits_test = fully_connected(
            inputs=concat0, out_dim=self.n_classes, scope_name='fc0')

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.label, logits=self.logits_train)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        with tf.name_scope('optimize'):
            _opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
            self.opt = _opt.minimize(self.loss, global_step=self.gstep)

    def summaries(self):
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

    def eval(self):
        with tf.name_scope('eval'):
            preds = tf.nn.softmax(self.logits_test)
            correct_preds = tf.equal(
                tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):

        self.import_data()
        self.get_embedding()
        self.model()
        self.loss()
        self.optimize()
        self.eval()
        self.summaries()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
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
                     self.train_summary_op])
                writer.add_summary(summaries, global_step=step)

                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step = step + 1
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, './checkpoints/tcNet_polarity/polarity_tcNet', step)

        print('\nAverage training loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Training accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_train))
        print('Took: {0} seconds'.format(time.time() - start_time))

        return step

    def eval_once(self, sess, init, writer, epoch, val_step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l, accuracy_batch, summaries = sess.run(
                    [self.loss, self.accuracy, self.val_summary_op])
                writer.add_summary(summaries, global_step=val_step)
                total_correct_preds = total_correct_preds + accuracy_batch
                total_loss = total_loss + l
                n_batches = n_batches + 1
                val_step = val_step + 1
        except tf.errors.OutOfRangeError:
            pass

        print('Average validation loss at epoch {0}: {1}'.format(
            epoch, total_loss / n_batches))
        print('Validation accuracy at epoch {0}: {1}'.format(
            epoch, total_correct_preds / self.n_test))
        print('Took: {0} seconds\n'.format(time.time() - start_time))

        return val_step

    def train(self, n_epochs):
        utils.mkdir_safe('.\\checkpoints')
        utils.mkdir_safe('.\\checkpoints\\tcNet_polarity')

        train_writer = tf.summary.FileWriter(
            '.\\graphs\\tcNet\\train', tf.get_default_graph())

        val_writer = tf.summary.FileWriter(
            '.\\graphs\\tcNet\\val')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state(os.path.dirname(
            #     '.\\checkpoints\\tcNet_polarity\\checkpoint'))

            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.gstep.eval()
            val_step = 0

            for epoch in range(n_epochs):
                step = self.train_one_epoch(
                    sess, saver, self.train_init, train_writer, epoch, step)
                val_step = self.eval_once(
                    sess, self.val_init, val_writer, epoch, val_step)

        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    model = TextConvNet()
    model.build()
    model.train(n_epochs=10)
