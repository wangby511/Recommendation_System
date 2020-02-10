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
#
# s = tf.constant([[[1.0,0,0,0],[1,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],
#                  [[1.0,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]]
#                 )
# # print(dice(x1))
# print("out = ",out)
# with tf.Session() as sess:
#     print("mean1 =", sess.run(mean1))
#     print("std1 =", sess.run(std1))
#     print("ps =", sess.run(ps))
#     # print("out =", sess.run(out))
#     # print("broadcast_shape = ", sess.run(str(broadcast_shape)))


