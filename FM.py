import numpy as np
import tensorflow as tf
import time

class FM():
    """
    FM 因子分解机
    特征交叉
    """

    def __init__(self,
                 feature_number,
                 embedding_size=10,
                 epochs=10,
                 batch_size=128,
                 learning_rate=0.005
                 ):
        self.p = feature_number
        self.k = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder('float', shape=[None, self.p])
            self.y = tf.placeholder('float', shape=[None, 1])

            # bias and weights
            w0 = tf.Variable(tf.zeros([1]))
            W = tf.Variable(tf.zeros([self.p]))

            # interaction factors, randomly initialized
            V = tf.Variable(tf.random_normal([self.k, self.p], stddev=0.01))

            # estimate of y, initialized to 0.
            # self.y_hat = tf.Variable(tf.zeros([None, 1]))

            linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, self.X), 1, keep_dims=True))
            pair_interactions = (
                tf.multiply(0.5, tf.reduce_sum(
                    tf.subtract(tf.pow(tf.matmul(self.X, tf.transpose(V)), 2),
                                tf.matmul(tf.pow(self.X, 2), tf.transpose(tf.pow(V, 2)))
                                ), 1, keep_dims=True)
                            )
            )
            self.y_hat = tf.add(linear_terms, pair_interactions)

            # L2 regularized sum of squares loss function over W and V
            lambda_w = tf.constant(0.001, name='lambda_w')
            lambda_v = tf.constant(0.001, name='lambda_v')

            l2_norm = tf.reduce_sum(
                tf.add(tf.multiply(lambda_w, tf.pow(W, 2)), tf.multiply(lambda_v, tf.pow(V, 2))))

            self.error = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_hat)))
            self.loss = tf.add(self.error, l2_norm)

            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def fit(self, X_train, y_train, X_valid, y_valid):
        n = X_train.shape[0]
        for epoch in range(self.epochs):
            perm = np.random.permutation(n)  # 将整个训练集(90570个)随机打乱顺序 n = X_train.shape[0]

            # iterate over batches
            X_PERM = X_train[perm]
            y_PERM = y_train[perm]
            for i in range(0, n, self.batch_size):
                loss, _ = self.sess.run([self.loss, self.optimizer],feed_dict={
                    self.X: X_PERM[i:i + self.batch_size - 1].reshape(-1, self.p),
                    self.y: y_PERM[i:i + self.batch_size - 1].reshape(-1, 1)
                })
            print("epoch =", epoch, ", loss = ", loss)

            if epoch % 10 != 0 or X_valid is None or y_valid is None:
                continue

            errors = self.sess.run(self.error, feed_dict={self.X: X_valid, self.y: y_valid.reshape(-1, 1)})
            RMSE = np.sqrt(np.array(errors).mean())
            print("epoch =", epoch, ", RMSE = ", RMSE)

