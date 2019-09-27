import time
import numpy as np
import tensorflow as tf


class NFM():
    """
    NFM
    神经网络因子分解机（Neural Factorization Machines，NFM）
    利用二阶交互池化层（Bi-Interaction Pooling）对FM嵌入后的向量两两进行元素级别的乘法，形成同维度的向量求和后作为前馈神经网络的输入。

    """

    def __init__(self,
                 field_size,
                 feature_size,
                 embedding_size=10,
                 batch_size=128,
                 learning_rate=0.001,
                 epoch=10
                 ):
        self.field_size = field_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # input for training
            # self.X = tf.placeholder('float', shape=[None, self.feature_size])
            self.feat_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name='feat_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

            # bias and feature bias
            self.bias = tf.Variable(tf.constant(0.1))
            self.feature_bias = tf.Variable(tf.random_normal([self.feature_size, 1], mean=0.0, stddev=0.01))

            # interaction factors, randomly initialized
            self.feature_embeddings = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], mean=0.0, stddev=0.01))

            # estimate of y, initialized to 0.
            # self.y_hat = tf.Variable(tf.zeros([None, 1]))

            # part I embeddings
            self.embedding_lookup = tf.nn.embedding_lookup(self.feature_embeddings, self.feat_index) # None * F * K
            self.feat_value_reshape = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # None * F * 1
            self.embeddings = tf.multiply(self.embedding_lookup, self.feat_value_reshape) # None * F * K

            # print("self.embeddings.get_shape() =",self.embeddings.get_shape()) # None * F * K

            # part I bias
            self.y_bias = self.bias * tf.ones_like(self.label) # None * 1
            # print("self.y_bias.get_shape() =", self.y_bias.get_shape())

            # part II first order term
            self.y_first_order = tf.nn.embedding_lookup(self.feature_bias, self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, self.feat_value_reshape), axis=2) # None * F

            # part IIIa second order term
            self.pair_interactions = 0.5 * tf.subtract(
                tf.square(tf.reduce_sum(self.embeddings, 1)),
                tf.reduce_sum(tf.square(self.embeddings),axis=1)
            )
            # None * K
            # print("self.pair_interactions.get_shape() =", self.pair_interactions.get_shape())

            # part IIIb second order term in deep layers

            self.weights_layer0 = tf.Variable(tf.random_normal([self.embedding_size, 32], mean=0.0, stddev=0.0))
            self.weights_layer0_bias = tf.Variable(tf.random_normal([1, 32], mean=0.0, stddev=1.0))

            self.weights_layer1 = tf.Variable(tf.random_normal([32, 32], mean=0.0, stddev=1.0))
            self.weights_layer1_bias = tf.Variable(tf.random_normal([1, 32], mean=0.0, stddev=1.0))


            self.pair_interaction_deep = tf.add(tf.matmul(self.pair_interactions, self.weights_layer0),
                                                self.weights_layer0_bias)
            self.pair_interaction_deep = tf.nn.relu(self.pair_interaction_deep)

            self.pair_interaction_deep = tf.add(tf.matmul(self.pair_interaction_deep, self.weights_layer1),
                                                self.weights_layer1_bias)
            self.pair_interaction_deep = tf.nn.relu(self.pair_interaction_deep)


            self.y_first_order = tf.reduce_sum(self.y_first_order, axis=1, keep_dims=True)

            self.linear_terms = tf.add(self.y_first_order, self.y_bias)

            self.pair_interaction_deep = tf.reduce_sum(self.pair_interaction_deep, axis=1, keep_dims=True)

            # print(self.y_bias.get_shape())
            # print(self.y_first_order.get_shape())
            # print(self.pair_interaction_deep.get_shape())

            self.y_hat = tf.add(self.pair_interaction_deep, self.linear_terms)

            # self.y_hat = tf.sigmoid(self.y_hat)

            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.label, self.y_hat)))

            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def fit(self,
            Xi_train,
            Xv_train,
            y_train,
            Xi_valid=None,
            Xv_valid=None,
            y_valid=None,
            early_stopping=False,
            refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        for epoch in range(self.epoch):
            t1 = time.time()
            total_batch = int(len(y_train) / self.batch_size)
            print("total_batch = ", total_batch) # 78
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                feed_dict = {
                    self.feat_index: Xi_batch,
                    self.feat_value: Xv_batch,
                    self.label: y_batch
                }
                loss, y_hat, opt = self.sess.run([self.loss, self.y_hat,self.optimizer], feed_dict=feed_dict)
                print("epoch:", epoch, ",i: ",i, " loss = ", loss)
                # print("self.y_hat",y_hat.reshape(1,-1))


            t2 = time.time()
            print("Total training time:",t2 - t1," s")
