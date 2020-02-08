import numpy as np
import tensorflow as tf

class AFM():
    """
    [AFM]
    """

    def __init__(self,
                 feature_size,
                 field_size,
                 embedding_size = 8,
                 attention_size = 10,
                 deep_layers=[32, 32],
                 deep_init_size = 50,
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.001,
                 random_seed=2020,
                 ):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.attention_size = attention_size
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

                # input for training
                self.feat_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name='feat_index')
                self.feat_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name='feat_value')
                self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

                # bias and feature bias
                self.bias = tf.Variable(tf.constant(0.1))
                self.feature_bias = tf.Variable(tf.random_normal([self.feature_size, 1], mean=0.0, stddev=0.01))

                # interaction factors, randomly initialized
                self.feature_embeddings = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], mean=0.0, stddev=0.01))

                # ATTENTION PART ! THAT IS NEW PART!
                # attention part
                glorot = np.sqrt(2.0 / (self.attention_size + self.embedding_size))

                self.weights_attention_w = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, self.attention_size)),dtype=tf.float32, name='attention_w')

                self.weights_attention_b = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.attention_size,)),dtype=tf.float32, name='attention_b')

                self.weights_attention_h = tf.Variable(
                    np.random.normal(loc=0, scale=1, size=(self.attention_size,)),dtype=tf.float32, name='attention_h')

                self.weights_attention_p = tf.Variable(np.ones((self.embedding_size, 1)), dtype=np.float32)

                # estimate of y, initialized to 0.
                # self.y_hat = tf.Variable(tf.zeros([None, 1]))


                # embeddings
                # None means batch_size!
                self.embedding_lookup = tf.nn.embedding_lookup(self.feature_embeddings, self.feat_index)  # None * F * K
                self.feat_value_reshape = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # None * F * 1
                self.embeddings = tf.multiply(self.embedding_lookup, self.feat_value_reshape)  # None * F * K

                # print("self.embeddings.get_shape() =",self.embeddings.get_shape()) # None * F * K

                # part I element_wise
                element_wise_product_list = []
                for i in range(self.field_size):
                    for j in range(i + 1, self.field_size):
                        element_wise_product_list.append(
                            tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))  # None * K

                self.element_wise_product = tf.stack(element_wise_product_list)  # (F * F - 1 / 2) * None * K
                self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1, 0, 2],
                                                         name='element_wise_product')  # None * (F * F - 1 / 2) *  K


                # part II self.interaction

                num_interactions = int(self.field_size * (self.field_size - 1) / 2)

                x = tf.reshape(self.element_wise_product,shape=(-1,self.embedding_size))
                wx_puls_b = tf.add(tf.matmul(x,self.weights_attention_w), self.weights_attention_b)
                self.attention_wx_plus_b = tf.reshape(wx_puls_b, shape=[-1, num_interactions, self.attention_size]) # N * ( F * [F - 1] / 2) * A

                h_relu_wx_plus_b = tf.multiply(tf.nn.relu(self.attention_wx_plus_b), self.weights_attention_h)
                self.attention_exp = tf.exp(tf.reduce_sum(h_relu_wx_plus_b, axis=2, keep_dims=True)) # N * ( F * F - 1 / 2) * 1


                self.attention_exp_sum = tf.reduce_sum(self.attention_exp, axis=1, keep_dims=True)  # N * 1 * 1

                self.attention_out = tf.div(self.attention_exp, self.attention_exp_sum, name='attention_out')  # N * ( F * F - 1 / 2) * 1

                self.attention_x_product = tf.reduce_sum(tf.multiply(self.attention_out, self.element_wise_product), axis=1, name='AFM')  # N * K

                self.attention_part_sum = tf.matmul(self.attention_x_product, self.weights['attention_p'])  # N * 1

                # part III. first order term
                self.y_first_order = tf.nn.embedding_lookup(self.feature_bias, self.feat_index)
                self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, self.feat_value), 2)

                # bias
                self.y_bias = self.bias * tf.ones_like(self.label)

                # out
                self.out = tf.add_n([tf.reduce_sum(self.y_first_order, axis=1, keep_dims=True),
                                     self.attention_part_sum,
                                     self.y_bias], name='out_afm')

                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

                # init
                self.saver = tf.train.Saver()
                init = tf.global_variables_initializer()
                self.sess = tf.Session()
                self.sess.run(init)
