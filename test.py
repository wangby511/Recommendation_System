# import tensorflow as tf
# import numpy as np
# with tf.Graph().as_default():
#     # a = tf.constant([1,2,3,4],name='a')
#     # b = tf.reshape(a,[-1,1])
#     # c = tf.concat([b,b],axis=1)
#     k = 3
#
#     params = tf.constant(np.random.rand(30, 4, 5, 3))
#     _, indices = tf.nn.top_k(params, k, sorted=False)
#     # tf.nn.top_k : 这个函数的作用是返回input中每行最大的k个数，并且返回它们所在位置的索引。
#     shape = tf.shape(indices)
#     r1 = tf.reshape(tf.range(shape[0]), [-1, 1])
#     r2 = tf.reshape(tf.range(shape[1]), [-1, 1])
#     r3 = tf.reshape(tf.range(shape[2]), [-1, 1])
#
#     r1 = tf.tile(r1, [1, shape[1] * shape[2] * k])
#     r2 = tf.tile(r2, [1, shape[2] * k])
#     r3 = tf.tile(r3, [1, k])
#
#     r2 = tf.reshape(r2, [-1, 1])
#     r3 = tf.reshape(r3, [-1, 1])
#
#     r1 = tf.reshape(r1, [-1, 1])
#     r2 = tf.tile(r2, [shape[0], 1])
#     r3 = tf.tile(r3, [shape[0] * shape[1], 1])
#
#     # indices = tf.reshape(indices, [-1, 1])
#
#     # indices = tf.tile(indices,[k, 1])
#
#     indices = tf.concat([r1, r2, r3, indices], 1)
#
#     shape = tf.shape(params)
#     flat = tf.reshape(params, [-1])
#     flat_idx = indices[:, 0] * shape[1] * shape[2] * shape[3] + indices[:, 1] * shape[2] * shape[3] + indices[:, 2] * shape[3] + indices[:, 3]
#
#     sess = tf.Session()
#     print(sess.run(shape))
#     print(sess.run(flat))
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# # calculate cross_entropy
# y = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
# y_ = tf.constant([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
# ysoft = tf.nn.softmax(y)
# cross_entropy = -tf.reduce_sum(y_ * tf.log(ysoft))
#
# # do cross_entropy just one step
# cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
#
# cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
#
# batch_mean, batch_variance = tf.nn.moments(y, [0])
#
# lrn = tf.nn.batch_normalization(y,mean=batch_mean,variance=batch_variance,scale=1,offset=0,variance_epsilon=1e-3)
#
# x = tf.constant([[1,2],[3,3]])
# y = tf.constant([[2,1],[4,6]])
# p1 = x * y
# p_multiply = tf.multiply(x,y)
# p_matmul = tf.matmul(x,y)
# p_substract = tf.subtract(x,y)
x1 = tf.constant([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
x2 = tf.constant([[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])

x3 = tf.constant([1,2,3,4,5])
x3 = tf.reshape(x3,(-1,1))
x4 = tf.constant([3])
x5 = tf.add(x3,x4)
x6 = tf.multiply(x1,x3)
# label1 = tf.argmax(x1, axis=1)
# label2 = tf.argmax(x2, axis=1)
# type = tf.cast(tf.equal(label1,label2), tf.int32)
# labels = tf.one_hot(type,2)
#
# x3 = tf.constant([3,2,1,0,1])
# y3 = tf.one_hot(x3, 4)
# x = tf.reduce_sum(x1,axis=0,keep_dims=True)
# y = tf.reduce_sum(x1,axis=0)



# with tf.compat.v1.Session() as sess:
    # print("label1=", sess.run(label1))
    # print("label2=", sess.run(label2))
    # print(sess.run(x3))
    # print(sess.run(x6))
    # print("y=", sess.run(y))
    # print("labels=",sess.run(labels))
#     print("p_multiply=", sess.run(p_multiply))
#     print("p_matmul=", sess.run(p_matmul))
#     # print(sess.run(ysoft))
#     # print("step2:cross_entropy result=")
#     # print(sess.run(cross_entropy))
#     # print("Function(softmax_cross_entropy_with_logits) result=")
#     # print(sess.run(cross_entropy2))
# print("cross_entropy_loss result=")

# a=np.array([[1,2,3],[4,5,6]])
# b=np.array([[7,8,9],[10,11,12]])
# d = np.concatenate((a,b),axis = -1)
# print(d)



TRAIN_FILE = "Driver_Prediction_Data/train.csv"
TEST_FILE = "Driver_Prediction_Data/test.csv"

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

from WideAndDeep import WideAndDeep
def main():
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
            feature_dict[col] = dict(zip(unique_val, range(total_feature, len(unique_val) + total_feature)))
            total_feature += len(unique_val)
            # print("total_feature =",total_feature,"\n")
    # print(total_feature)  # 254

    """
    对训练集进行转化
    """
    train_y = dfTrain[['target']].values.tolist()
    dfTrain.drop(['target', 'id'], axis=1, inplace=True)
    train_feature_index = dfTrain.copy()
    train_feature_value = dfTrain.copy()
    # n_train = train_feature_value.shape[0]

    for col in train_feature_index.columns:
        if col in IGNORE_COLS:
            train_feature_index.drop(col, axis=1, inplace=True)
            train_feature_value.drop(col, axis=1, inplace=True)
            continue
        elif col in NUMERIC_COLS:
            train_feature_index[col] = feature_dict[col]
        else:
            train_feature_index[col] = train_feature_index[col].map(feature_dict[col])
            train_feature_value[col] = 1
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


    wideanddeep = WideAndDeep(feature_size=total_feature, field_size=field_size, embedding_size=8)

    """train"""
    wideanddeep.train(train_feature_index, train_feature_value, train_y)
    # epoch 199, loss is 0.15737674

main()
