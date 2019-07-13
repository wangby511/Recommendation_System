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
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.001,
                 random_seed=2019,
                 ):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = random_seed

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

            # Quardatic Part
            """
            IPNN
            
            f_i ...  K * 1 维
            f = (f_1^T, f_2^T, ..., f_N^T) 
            N = field_size
            lp_I = sigma(i = 1...N)sigma(j = 1...N)w_i * w_j * (f_i1 * f_j1 + ... + f_iK * f_jK)
            = sigma(i = 1...N)(w_i * f_i1)sigma(j = 1...N)(w_j * f_j1) + ... + sigma(i = 1...N)(w_i * f_iK)sigma(j = 1...N)(w_j * f_jK)
            
            let f_column_k =  sigma(i = 1...N)(w_i * f_ik)
            lp_I = (f_column_1)^2 + (f_column_2)^2 + ... + (f_column_K)^2
            = [f_column_1,f_column_2,...,f_column_K]的第二范式，平方和
            
            lp = [lp_1,lp_2,...,lp_I,...,lp_D1] (D1 = init_deep_size = 50)
            
            self.lp = BATCH_SIZE * D1 维度
            
            TIME COMPLEXITY:
            p_ij :M
            p : M * N * N
            lp : N * N * D1 -> calculate the sum for each p matrix
            
            TOTAL for single sample: (M + D1) * N * N also (K + D1) * F * F, F = field_size = N, K = embedding_size = M
            But in the calculation: 
            N * M * D1 also F * K * D1
            
            """
            quadratic_output = []

            for i in range(self.deep_init_size):

                weight = tf.reshape(self.weights['product-quadratic-inner'][i], (1, -1, 1)) # 1 x F x 1 (2019.7.8)

                f_segma = tf.reduce_sum(tf.multiply(self.embeddings, weight),axis=1)  # N * K

                lp_i = tf.reshape(tf.norm(f_segma, axis = 1), shape=(-1, 1)) # (2019.7.9)

                quadratic_output.append(lp_i)  # N * 1

            self.lp = tf.concat(quadratic_output, axis=1)  # N * init_deep_size


            """
            OPNN
            p_ij = f_i * (f_j)^T ... K * K 维
            pij_ab = fi_a * fj_b
            sigma(i = 1...F)sigma(j = 1...F) pij_ab 
            = sigma(i = 1...F)(fi_a) * sigma(j = 1...F)(fj_b)
            = sigma(i = 1...F)(fi)在a,b两个index的乘积
            f_sigma = sigma(i = 1...F)(fi) ... M * 1 维
            
            P_ab = f_sigma[a] * f_sigma[b] (a,b = 1,...,K)
            OUT_ab_i = f_sigma[a] * f_sigma[b] * parameter_i[a,b] (a,b = 1,...,K)
            
            OUT_ab = [OUT_ab_1, ..., OUT_ab_j] (j = 1,...,50) in this example 
            
            parameter = [50 * K * K] in demension
            
            TIME COMPLEXITY:
            p_ij : M * M
            p : M * M * N * N
            lp : M * M * N * N * D2
            
            TOTAL for single sample: M^2 * N^2 * D2 also K^2 * F^2 * D2, F = field_size = N, K = embedding_size = M
            But in the calculation: 
            (N * M + M * M) * D2 = (N + M) * M * D2 also (F + K)* K * D2
            """
            """
            quadratic_output = []
            
            for i in range(self.deep_init_size):

            f_sigma = tf.reduce_sum(self.embeddings, axis = 1)
            
            p = tf.matmul(tf.expand_dims(f_sigma,2),tf.expand_dims(f_sigma,1)) # N * K * K = (N * K * 1) * (N * 1 * K)
            
            weights['product-quadratic-outer'] = tf.Variable(
                tf.random_normal([self.deep_init_size, self.embedding_size,self.embedding_size], 0.0, 0.01))
            
            for i in range(self.deep_init_size):
            
                theta = tf.multiply(p,tf.expand_dims(self.weights['product-quadratic-outer'][i],0)) # N * K * K
                
                lp_i = tf.reshape(tf.reduce_sum(theta,axis=[1,2]),shape=(-1,1)) # N * 1
                
                quadratic_output.append(lp_i) 
                
            """

            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))

            # Deep layer0
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_0"]), self.weights["bias_0"])
            self.y_deep = tf.nn.relu(self.y_deep)

            # Deep layer1
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_1"]), self.weights["bias_1"])
            self.y_deep = tf.nn.relu(self.y_deep)

            self.out = tf.add(tf.matmul(self.y_deep, self.weights['output']), self.weights['output_bias'])

            # self.loss_type == "logloss":
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)

            # self.loss_type == "mse":
            # self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9,
                                                    beta2=0.999,
                                                    epsilon=1e-8
                                                    ).minimize(self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)


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
