import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
# 2020 Feb 8th

class WideAndDeep(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_size=100,
                 field_size=39,
                 embedding_size=8,
                 deep_layers=[512, 256, 1],
                 # deep_layers=[1024, 512, 256, 1],
                 batch_size=256,
                 learning_rate=0.001,
                 optimizer="adam",
                 random_seed=2020,
                 l2_reg=0.0):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.l2_reg = l2_reg

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.random_seed = random_seed
        self.train_result, self.valid_result = [],[]
        self.max_iteration = 100

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            # self.weights = self._initialize_weights()
            self.weights = dict()

            # initialize embedding layer
            self.weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')

            self.weights['wide_weight'] = tf.Variable(tf.random_normal([self.field_size * self.embedding_size], 0.0, 1.0), name='wide_weight')
            self.weights['wide_bias'] = tf.Variable(tf.random_normal([1, 1], 0.0, 1.0), name='wide_weight')

            # initialize deep layers 1-3
            input_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
            self.weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
            self.weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)

            glorot = np.sqrt(2.0 / (self.deep_layers[0] + self.deep_layers[1]))
            self.weights["layer_1"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[0], self.deep_layers[1])), dtype=np.float32)
            self.weights["bias_1"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[1])), dtype=np.float32)

            glorot = np.sqrt(2.0 / (self.deep_layers[1] + self.deep_layers[2]))
            self.weights["layer_2"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[1], self.deep_layers[2])),dtype=np.float32)
            self.weights["bias_2"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[2])), dtype=np.float32)

            # final concat projection layer
            # input_size = self.field_size + self.embedding_size + self.deep_layers[1]

            # glorot = np.sqrt(2.0 / (input_size + 1))
            # self.weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
            # self.weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

            # Model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)  # N * F * K
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
            self.embeddings = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # (N) * (F * K)

            # Wide Part
            self.y_wide1 = tf.multiply(self.embeddings, self.weights['wide_weight'])
            self.y_wide1 = tf.add(self.y_wide1, self.weights['wide_bias'])
            self.y_wide = tf.reshape(tf.reduce_sum(self.y_wide1, 1), shape=[-1,1])


            # Deep Part
            self.y_deep = tf.add(tf.matmul(self.embeddings, self.weights["layer_0"]), self.weights["bias_0"])
            self.y_deep = tf.nn.relu(self.y_deep)

            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_1"]), self.weights["bias_1"])
            self.y_deep = tf.nn.relu(self.y_deep)

            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_2"]), self.weights["bias_2"])
            self.y_deep = tf.nn.relu(self.y_deep)

            # Combine Wide And Deep Part Together by different weights
            self.weights['wide_weight'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
            self.weights['deep_weight'] = tf.Variable(tf.constant(1.00), dtype=np.float32)
            self.out = tf.add(tf.multiply(self.weights['wide_weight'], self.y_wide),
                              tf.multiply(self.weights['deep_weight'], self.y_deep))

            # Loss Part
            # if self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
            # elif self.loss_type == "mse":
            # self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # # l2 regularization on weights
            # if self.l2_reg > 0:
            #     self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
            #     if self.use_deep:
            #         self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_0"])
            #         self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_1"])

            # Optimizer Part
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8
            ).minimize(self.loss)

            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate,
                    initial_accumulator_value=1e-8
                ).minimize(self.loss)

            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init Session
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def train(self, train_feature_index, train_feature_value, train_y):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.max_iteration):
                epoch_loss, _ = sess.run([self.loss, self.optimizer],
                                         feed_dict={self.feat_index: train_feature_index,
                                                    self.feat_value: train_feature_value,
                                                    self.label: train_y}
                                         )
                print("epoch %s,loss is %s" % (str(i), str(epoch_loss)))

