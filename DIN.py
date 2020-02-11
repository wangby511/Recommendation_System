import numpy as np
import tensorflow as tf
# https://zhuanlan.zhihu.com/p/42934748
# 推荐系统之阿里广告：Deep Interest Network for Click-Through Rate Predictioin - DIN

# 数据为核心的自适应激活函数Data Adaptive Activation Function
def dice(s, axis=-1, epsilon=1e-8, name=''):

    alpha = tf.compat.v1.get_variable('alpha' + name, s.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    input_shape = list(s.get_shape())
    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]

    broadcast_shape = [1] * len(input_shape) # [1,1,...,1,1]
    broadcast_shape[axis] = int(input_shape[axis]) # [1,1,...,1,hidden_unit_size] if axis == -1

    mean = tf.reduce_mean(s, axis=reduction_axes)
    mean = tf.reshape(mean, broadcast_shape)

    std = tf.sqrt(tf.reduce_mean(tf.square(s - mean) + epsilon, axis=reduction_axes))
    std = tf.reshape(std, broadcast_shape)

    # s_normalized = (s - mean) / (std + epsilon)
    # s_normed = tf.layers.batch_normalization(s, center=False, scale=False)  # a simple way to use BN to calculate x_p
    ps = tf.sigmoid((s - mean) / (std + epsilon))

    dice_out = alpha * (1.0 - ps) * s + ps * s

    return dice_out


def attention(queries, keys, keys_length):
    '''
      queries:     [B, H]   item_emb
      keys:        [B, T, H]   hist_emb
      keys_length: [B]    sample_len
    '''
    queries_hidden_units = queries.get_shape().as_list()[-1]
    # H

    keys_time_units = tf.shape(keys)[1]

    queries = tf.tile(queries, [1, keys_time_units])
    # shape: [B, H * T]

    queries = tf.reshape(queries, [-1, keys_time_units, queries_hidden_units])
    # shape: [B, T, H]

    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
    # shape: [B, T, 4H]

    d_layer_1_all = tf.layers.dense(
        din_all,
        80,
        activation=tf.nn.sigmoid,
        name='f1_atttention',
        reuse=tf.AUTO_REUSE

    )
    d_layer_2_all = tf.layers.dense(
        d_layer_1_all,
        40,
        activation=tf.nn.sigmoid,
        name='f2_attention',
        reuse=tf.AUTO_REUSE
    )
    d_layer_3_all = tf.layers.dense(
        d_layer_2_all,
        1,
        activation=None,
        name='f3_atttention',
        reuse=tf.AUTO_REUSE
    )

    outputs = tf.reshape(d_layer_3_all, [-1, 1, keys_time_units]) # [B, 1, T]

    key_masks = tf.sequence_mask(keys_length, keys_time_units)  # [keys_length, T] [keys_length * True + (T - keys_length) * False]

    key_masks = tf.expand_dims(key_masks, 1)

    paddings = tf.ones_like(outputs) # [32, 1, T]

    outputs = tf.where(key_masks, outputs, paddings) # [32, 1, T]

    # Scale  [32, 1, T] / sqrt(128)
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum 加权平均 三维矩阵相乘，相乘发生在后两维 [B, 1, T] * [B, T, H] = [B, 1, H]
    outputs = tf.matmul(outputs, keys)

    return outputs

#
# a = tf.expand_dims(tf.sequence_mask(5, 10), 1)
# a1 = tf.constant([[10],[20],[30],[40],[50],[60],[70],[80],[90],[100]], tf.int32)
# a2 = tf.constant([[15],[25],[35],[45],[55],[65],[75],[85],[95],[105]], tf.int32)
# b = tf.constant([1,3], tf.int32)
# c = tf.tile(a, b)
# c = tf.where(a, a1, a2)
#
# key_masks = tf.sequence_mask(5,10)
# key_masks = tf.expand_dims(key_masks, 1)
#
# with tf.Session() as sess:
#     print("c =", sess.run(c))
#     # print("paddings = ",sess.run(paddings))
#
# print((-2 ** 32 + 1))
# s = tf.constant([[[1.0,0,0,0],[1,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],
#                  [[1.0,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]]
#                 )
# print("out = ",out)
# with tf.Session() as sess:
#     print("mean1 =", sess.run(mean1))
#     print("std1 =", sess.run(std1))
#     print("ps =", sess.run(ps))
#     # print("out =", sess.run(out))
#     # print("broadcast_shape = ", sess.run(str(broadcast_shape)))


