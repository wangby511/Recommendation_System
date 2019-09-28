import time
import numpy as np
import tensorflow as tf


class DCN():
    """
    DCN
    2019 09 27
    Deep&Cross Network模型我们下面将简称DCN模型：
    一个DCN模型从嵌入和堆积层开始，接着是一个交叉网络和一个与之平行的深度网络，之后是最后的组合层，它结合了两个网络的输出。

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

            self.embeddings = tf.reshape(self.embeddings,shape=[-1, self.field_size * self.embedding_size])
            print("self.embeddings.get_shape() =",self.embeddings.get_shape()) # None * (F * K)

            ### Deep部分就是普通的MLP网络，主要是全连接。
            # part II Deep Part

            self.dot_size = self.field_size * self.embedding_size

            self.weights_deep_layer0 = tf.Variable(tf.random_normal([self.dot_size, 32], mean=0.0, stddev=0.01))
            self.weights_deep_layer0_bias = tf.Variable(tf.random_normal([1, 32], mean=0.0, stddev=0.01))

            self.weights_deep_layer1 = tf.Variable(tf.random_normal([32, 32], mean=0.0, stddev=0.01))
            self.weights_deep_layer1_bias = tf.Variable(tf.random_normal([1, 32], mean=0.0, stddev=0.01))


            self.deep = tf.add(tf.matmul(self.embeddings, self.weights_deep_layer0), self.weights_deep_layer0_bias)
            self.deep = tf.nn.relu(self.deep)

            self.deep = tf.add(tf.matmul(self.deep, self.weights_deep_layer1), self.weights_deep_layer1_bias)
            self.deep = tf.nn.relu(self.deep)
            print("self.deep.get_shape() = ",self.deep.get_shape()) # None * 32

            ### Cross部分是对FM部分的推广。
            ### x(l + 1) = x0 * x(l)^T * w(l) + b(l) + x(l)
            # part III Cross Part

            self.weights_cross_layer0 = tf.Variable(tf.random_normal([self.dot_size, 1], mean=0.0, stddev=0.01))
            self.weights_cross_layer0_bias = tf.Variable(tf.random_normal([self.dot_size,1], mean=0.0, stddev=0.01))

            self.weights_cross_layer1 = tf.Variable(tf.random_normal([self.dot_size, 1], mean=0.0, stddev=0.01))
            self.weights_cross_layer1_bias = tf.Variable(tf.random_normal([self.dot_size,1], mean=0.0, stddev=0.01))

            self.x0 = tf.reshape(self.embeddings, (-1, self.dot_size, 1))
            # None * D * 1, D = field_size * embedding_size

            self.x_l = self.x0

            ### cross layer I
            ###
            self.cross_part = tf.matmul(self.x0, self.x_l, transpose_b=True)
            print("self.cross_part.get_shape() = ", self.cross_part.get_shape())  # None * D * D

            # self.temp = tf.matmul(self.cross_part, self.weights_cross_layer0)
            # print("self.temp.get_shape() = ", self.temp.get_shape())
            # [None * D * D] * [D * 1] = [None * D * 1]

            self.x_l = tf.matmul(self.cross_part, self.weights_cross_layer0) + self. weights_cross_layer0_bias + self.x_l
            print("self.x_l.get_shape() = ", self.x_l.get_shape())

            ### cross layer II
            ###
            self.cross_part = tf.matmul(self.x0, self.x_l, transpose_b=True)
            print("self.cross_part.get_shape() = ", self.cross_part.get_shape())  # None * D * D

            self.x_l = tf.matmul(self.cross_part, self.weights_cross_layer1) + self.weights_cross_layer1_bias + self.x_l
            print("self.x_l.get_shape() = ", self.x_l.get_shape())

            self.x_l = tf.reshape(self.x_l, (-1, self.dot_size))
            print("self.x_l.get_shape() = ", self.x_l.get_shape()) # None * D

            self.out = tf.concat([self.deep, self.x_l], axis=1)
            print("self.out.get_shape() = ", self.out.get_shape())  # None * (field_size * embedding_size + 32)

            self.final_size = int(self.out.get_shape()[1])
            print("self.final_size = ", self.final_size)

            self.weights_concat_layer = tf.Variable(tf.random_normal([self.final_size, 1], mean=0.0, stddev=0.01))
            self.weights_concat_layer_bias = tf.Variable(tf.constant(0.01),dtype=np.float32)

            ###
            # Final output layer

            self.y_hat = tf.add(tf.matmul(self.out, self.weights_concat_layer), self.weights_concat_layer_bias)
            #
            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.label, self.y_hat)))
            #
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
