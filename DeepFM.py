import numpy as np
import tensorflow as tf
import pandas as pd
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_size=100,
                 field_size=39,
                 embedding_size=8,
                 deep_layers=[32, 32],
                 batch_size=256,
                 learning_rate=0.001,
                 optimizer="adam",
                 random_seed=2019,
                 use_fm=True,
                 use_deep=True,
                 loss_type="logloss",
                 l2_reg=0.0):

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_layers_activation = tf.nn.relu
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.random_seed = random_seed
        self.loss_type = loss_type
        self.train_result,self.valid_result = [],[]
        self.max_iteration = 200

        self._init_graph()

    def _initialize_weights(self):
        weights = dict()

        # embeddings layer
        weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        weights['feature_bias']       = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        # deep layers
        # num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights['bias_0']  = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)

        glorot = np.sqrt(2.0 / (self.deep_layers[0] + self.deep_layers[1]))

        weights["layer_1"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[0], self.deep_layers[1])), dtype=np.float32)
        weights["bias_1"]  = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[1])), dtype=np.float32)


        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[1] #[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[1]

        glorot = np.sqrt(2.0 / (input_size + 1))

        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)  # (F * K) * N
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # first order term
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)

            # second order term
            # sum-square-part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * k
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # squre-sum-part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)

            # Deep component
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])

            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_0"]), self.weights["bias_0"])
            self.y_deep = self.deep_layers_activation(self.y_deep)

            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_1"]), self.weights["bias_1"])
            self.y_deep = self.deep_layers_activation(self.y_deep)
            # y_second_order: None * 256, y_first_order: None * 39, y_deep: None * 32



            # ----DeepFM---------
            # if self.use_fm and self.use_deep:
            self.concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            # elif self.use_fm:
            #     concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            # elif self.use_deep:
            #     concat_input = self.y_deep

            self.out = tf.add(tf.matmul(self.concat_input, self.weights['concat_projection']), self.weights['concat_bias'])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_0"])
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_1"])

            # optimize
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

            # init
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


TRAIN_FILE = "Data/Driver_Prediction_Data/train.csv"
TEST_FILE = "Data/Driver_Prediction_Data/test.csv"

NUMERIC_COLS = [
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"
]

IGNORE_COLS = [
    "id", "target",
    "ps_calc_01", "ps_calc_02", "ps_calc_03", "ps_calc_04",
    "ps_calc_05", "ps_calc_06", "ps_calc_07", "ps_calc_08",
    "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12",
    "ps_calc_13", "ps_calc_14",
    "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin"
]


def main():
    print("[START]DeepFM PIPELINE BEGINS")
    dfTrain = pd.read_csv(TRAIN_FILE)  # shape (10000, 59)
    dfTest = pd.read_csv(TEST_FILE)  # shape (2000, 58)
    df = pd.concat([dfTrain, dfTest])
    # print (df.shape) # shape (12000, 59)

    feature_dict = {}
    total_feature = 0
    for col in df.columns:
        if col in IGNORE_COLS:
            continue
        elif col in NUMERIC_COLS:
            feature_dict[col] = total_feature
            total_feature += 1
        else:
            unique_val = df[col].unique()
            # print("unique_val =",unique_val)
            # print(dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature))))
            feature_dict[col] = dict(zip(unique_val, range(total_feature, len(unique_val) + total_feature)))
            total_feature += len(unique_val)
            # print("total_feature =",total_feature,"\n")
    # print(total_feature)  # 254
    # print(feature_dict)
    # 254
    # {'ps_car_01_cat': {10: 0, 11: 1, 7: 2, 6: 3, 9: 4, 5: 5, 4: 6, 8: 7, 3: 8, 0: 9, 2: 10, 1: 11, -1: 12}, ...

    """
    对训练集进行转化
    """
    train_y = dfTrain[['target']].values.tolist()
    dfTrain.drop(['target', 'id'], axis=1, inplace=True)
    train_feature_index = dfTrain.copy()
    train_feature_value = dfTrain.copy()
    n_train = train_feature_value.shape[0]

    for col in train_feature_index.columns:
        # print("col=",col)
        if col in IGNORE_COLS:
            train_feature_index.drop(col, axis=1, inplace=True)
            train_feature_value.drop(col, axis=1, inplace=True)
            continue
        elif col in NUMERIC_COLS:
            train_feature_index[col] = feature_dict[col]
        else:
            train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
            train_feature_value[col] = 1

    # print("train_feature_index =",train_feature_index)
    # print("train_feature_index =", train_feature_index.shape) #(N, 37)
    # print("train_feature_value =",train_feature_value)
    # print("train_feature_value =", train_feature_value.shape) #(N, 37)
    field_size = train_feature_value.shape[1]

    """
    对测试集进行转化
    """
    # test_ids = dfTest['id'].values.tolist()
    # dfTest.drop(['id'], axis=1, inplace=True)
    # test_feature_index = dfTest.copy()
    # test_feature_value = dfTest.copy()
    #
    # for col in test_feature_index.columns:
    #     if col in IGNORE_COLS:
    #         test_feature_index.drop(col, axis=1, inplace=True)
    #         test_feature_value.drop(col, axis=1, inplace=True)
    #         continue
    #     elif col in NUMERIC_COLS:
    #         test_feature_index[col] = feature_dict[col]
    #     else:
    #         test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
    #         test_feature_value[col] = 1


    deepfm = DeepFM(feature_size=total_feature, field_size=field_size, embedding_size=8)


    """train"""
    deepfm.train(train_feature_index, train_feature_value, train_y)
    # epoch 199, loss is 0.15737674


main()
