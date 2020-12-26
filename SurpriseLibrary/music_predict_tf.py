from __future__ import absolute_import, division, print_function
import pandas as pd
import time
import tensorflow as tf
import numpy as np

# 一批数据的大小
BATCH_SIZE = 2000

# 用户数
USER_NUM = 6040

# 电影数
ITEM_NUM = 3952

# factor维度
DIM = 15

# 最大迭代轮数
EPOCH_MAX = 100

np.random.seed(13575)

def read_data_and_process(filname):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep="::", header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df

# 调用上面的函数获取数据
def get_data():
    df = read_data_and_process("../Data/MovieLens/ml-1m/ratings.dat")
    rows = len(df) # rows = 1000209
    # df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index = int(rows * 0.9)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    # print(df_train.shape, df_test.shape)
    # (900188, 4)(100021, 4)
    return df_train, df_test

def clip(x):
    return np.clip(x, 1.0, 5.0)

def get_batch_size_data(data, batch_size, index):
    num_rows = data.shape[0]
    num_cols = data.shape[1]

    startIndex = index * batch_size
    endIndex = min((index + 1) * batch_size - 1, num_rows)
    out = data[startIndex:endIndex,:]
    # print(startIndex,endIndex)
    # print(out.shape) (2000, 3)
    returndata = [out[:, i] for i in range(num_cols)]
    return returndata


# 实际训练过程
def svd(train, test):
    TRAIN_SIZE = len(train)
    print ('TRAIN_SIZE =',TRAIN_SIZE) #TRAIN_SIZE = 900188

    # 一批一批数据用于训练
    training_data = np.transpose(np.array([train["user"], train["item"], train["rate"]]))
    testing_data = np.transpose(np.array([test["user"], test["item"], test["rate"]]))
    # print(training_data.shape)
    # (900188, 3)
    # print(testing_data.shape)
    # (100021, 3)

    ### 构建graph和训练
    w_user = tf.get_variable("embd_user", shape=[USER_NUM, DIM], initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_item = tf.get_variable("embd_item", shape=[USER_NUM, DIM], initializer=tf.truncated_normal_initializer(stddev=0.02))

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    global_bias = tf.get_variable("global_bias", shape=[])
    w_bias_user = tf.get_variable("embd_bias_user", shape=[USER_NUM])
    w_bias_item = tf.get_variable("embd_bias_item", shape=[USER_NUM])

    # bias向量
    bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
    bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")

    # user向量与item向量
    embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
    embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

    infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)

    # 加上几个偏置项
    infer = tf.add(infer, global_bias)

    # 加上正则化项
    regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")

    learning_rate = 0.001
    reg = 0.1
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))

    # global_step = tf.contrib.framework.get_or_create_global_step()
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 初始化所有变量
    init_op = tf.global_variables_initializer()

    # 开始迭代
    with tf.Session() as sess:
        sess.run(init_op)
        print("{} {} {} {}".format("epoch", "train_error", "eval_error", "time"))
        start = time.time()
        for i in range(EPOCH_MAX):
            errors = []

            for j in range(int(TRAIN_SIZE/BATCH_SIZE + 1)):
                users, items, rates = get_batch_size_data(training_data, BATCH_SIZE, j)

                _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users, item_batch: items, rate_batch: rates})
                pred_batch = clip(pred_batch)
                errors.append(np.mean(np.power(pred_batch - rates, 2)))

            # samples_per_batch = 900188
            train_err = np.mean(np.array(errors))
            users, items, rates = [testing_data[:, i] for i in range(testing_data.shape[1])]

            pred_batch = sess.run(infer, feed_dict={user_batch: users, item_batch: items})
            pred_batch = clip(pred_batch)
            test_err = np.mean(np.power(pred_batch - rates, 2))

            end = time.time()
            print("iter =",i,",train_err =",train_err,",test_err =",test_err,",time =",end - start,"s")
            start = end

def main():
    # 获取数据
    df_train, df_test = get_data()
    svd(df_train, df_test)

if __name__ == '__main__':
    main()