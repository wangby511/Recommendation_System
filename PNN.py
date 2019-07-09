# https://github.com/princewen/tensorflow_practice/blob/master/recommendation/Basic-PNN-Demo/PNN.py
import numpy as np
import tensorflow as tf

from time import time
from sklearn.base import BaseEstimator, TransformerMixin

class PNN(BaseEstimator, TransformerMixin):

    def __init__(self,
                 feature_size,
                 field_size,
                 embedding_size=8,
                 deep_layers=[32, 32],
                 deep_init_size = 50,
                 deep_layer_activation=tf.nn.relu,
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.001,
                 optimizer="adam",
                 verbose=False,
                 random_seed=2019,
                 loss_type="logloss",
                 ):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # Embeddings
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)  # N * F * K

            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])

            self.embeddings = tf.multiply(self.embeddings, feat_value)  # N * F * K

            # Linear Part
            linear_output = []
            for i in range(self.deep_init_size): # 50

                lz_i = tf.reduce_sum(tf.multiply(self.embeddings, self.weights['product-linear'][i]), axis=[1, 2])

                linear_output.append(tf.reshape(lz_i, shape=(-1,1)))  # N * 1

            self.lz = tf.concat(linear_output, axis=1)  # N * init_deep_size

            # Quardatic
            quadratic_output = []

            for i in range(self.deep_init_size):

                weight = tf.reshape(self.weights['product-quadratic-inner'][i], (1, -1, 1)) # 1 x F x 1

                f_segma = tf.reduce_sum(tf.multiply(self.embeddings, weight),axis=1)  # N * F * K

                quadratic_output.append(tf.reshape(tf.norm(f_segma), shape=(-1, 1)))  # N * 1

            self.lp = tf.concat(quadratic_output, axis=1)  # N * init_deep_size

            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))


    def _initialize_weights(self):

        weights = dict()

        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name='feature_embeddings')

        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        weights['product-quadratic-inner'] = tf.Variable(
            tf.random_normal([self.deep_init_size, self.field_size], 0.0, 0.01))

        weights['product-linear'] = tf.Variable(
            tf.random_normal([self.deep_init_size, self.field_size, self.embedding_size], 0.0, 0.01))

        weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size, ], 0, 0, 1.0))

        # deep layers
        input_size = self.deep_init_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        weights['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        glorot = np.sqrt(2.0 / (self.deep_layers[0] + self.deep_layers[1]))
        weights["layer_1"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[0], self.deep_layers[1])),
            dtype=np.float32)

        weights["bias_1"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[1])),
            dtype=np.float32)

        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['output'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[-1], 1)),
                                        dtype=np.float32)

        weights['output_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


x = PNN(254,37)
