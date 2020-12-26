import pandas as pd

TRAIN_FILE = "../Data/Driver_Prediction_Data/train.csv"
TEST_FILE = "../Data/Driver_Prediction_Data/test.csv"

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

dfTrain = pd.read_csv(TRAIN_FILE) # shape (10000, 59)
dfTest = pd.read_csv(TEST_FILE) # shape (2000, 58)
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
        feature_dict[col] = dict(zip(unique_val,range(total_feature,len(unique_val) + total_feature)))
        total_feature += len(unique_val)
        # print("total_feature =",total_feature,"\n")
print(total_feature) #254
print(feature_dict)

"""
对训练集进行转化
"""
# print(dfTrain.columns)
train_y = dfTrain[['target']].values.tolist()
# print("train_y =",len(train_y)) # 10000
dfTrain.drop(['target','id'],axis=1,inplace=True)
train_feature_index = dfTrain.copy()
train_feature_value = dfTrain.copy()

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
print("train_feature_index =",train_feature_index.shape)
# print("train_feature_value =",train_feature_value)
print("train_feature_value =",train_feature_value.shape)


"""
对测试集进行转化
"""
print(dfTest.columns)
"""
对测试集进行转化
"""
test_ids = dfTest['id'].values.tolist()
dfTest.drop(['id'],axis=1,inplace=True)

test_feature_index = dfTest.copy()
test_feature_value = dfTest.copy()

for col in test_feature_index.columns:
    if col in IGNORE_COLS:
        test_feature_index.drop(col,axis=1,inplace=True)
        test_feature_value.drop(col,axis=1,inplace=True)
        continue
    elif col in NUMERIC_COLS:
        test_feature_index[col] = feature_dict[col]
    else:
        test_feature_index[col] = test_feature_index[col].map(feature_dict[col])
        test_feature_value[col] = 1

import tensorflow as tf
import numpy as np
"""模型参数"""
dfm_params = {
    "use_fm":True,
    "use_deep":True,
    "embedding_size":8,
    "dropout_fm":[1.0,1.0],
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "l2_reg":0.01,
    "verbose":True,
    "eval_metric":'gini_norm',
    "random_seed":3
}
dfm_params['feature_size'] = total_feature
dfm_params['field_size'] = len(train_feature_index.columns)
print(dfm_params['field_size']) #37

feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

"""weights"""
weights = dict()

#embeddings
weights['feature_embeddings'] = tf.Variable(tf.random_normal([total_feature,dfm_params['embedding_size']],0.0,0.01),name='feature_embeddings')
# 254 x 8
weights['feature_bias'] = tf.Variable(tf.random_normal([total_feature,1],0.0,1.0),name='feature_bias')
# 254 x 1
"""embedding"""
embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'],feat_index) #N x 37 x 8

reshaped_feat_value = tf.reshape(feat_value,shape=[-1,dfm_params['field_size'],1]) # N x 37 x 1

embeddings = tf.multiply(embeddings,reshaped_feat_value) #N x 37 x 8

"""deep layers"""
num_layer = len(dfm_params['deep_layers'])
input_size = dfm_params['field_size'] * dfm_params['embedding_size'] # 37 x 8 = 296

glorot = np.sqrt(2.0/(input_size + dfm_params['deep_layers'][0]))

weights['layer_0'] = tf.Variable(
    np.random.normal(loc=0,scale=glorot,size=(input_size,dfm_params['deep_layers'][0])),dtype=np.float32)
# 296 x 32
weights['bias_0'] = tf.Variable(
    np.random.normal(loc=0,scale=glorot,size=(1, dfm_params['deep_layers'][0])),dtype=np.float32)
# 1 x 32

glorot = np.sqrt(2.0 / (dfm_params['deep_layers'][0] + dfm_params['deep_layers'][1]))
weights["layer_1"] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(dfm_params['deep_layers'][0], dfm_params['deep_layers'][1])),dtype=np.float32)
# 32 x 32
weights["bias_1"] = tf.Variable(
    np.random.normal(loc=0, scale=glorot, size=(1, dfm_params['deep_layers'][1])),dtype=np.float32)
# 1 x 32

input_size = dfm_params['field_size'] + dfm_params['embedding_size'] + dfm_params['deep_layers'][-1]
# 37 + 8 + 32 = 77
glorot = np.sqrt(2.0/(input_size + 1))
weights['concat_projection'] = tf.Variable(
    np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)


"""fm part"""
fm_first_order = tf.nn.embedding_lookup(weights['feature_embeddings'], feat_index)
fm_first_order = tf.reduce_sum(tf.multiply(fm_first_order, reshaped_feat_value), 2)

summed_features_emb = tf.reduce_sum(embeddings, 1)
summed_features_emb_square = tf.square(summed_features_emb)

squared_features_emb = tf.square(embeddings)
squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

fm_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

"""deep part"""
y_deep = tf.reshape(embeddings, shape=[-1, dfm_params['field_size'] * dfm_params['embedding_size']]) # 10000 x 296
y_deep = tf.add(tf.matmul(y_deep, weights["layer_0"]), weights["bias_0"])
y_deep = tf.nn.relu(y_deep)
y_deep = tf.add(tf.matmul(y_deep, weights["layer_1"]), weights["bias_1"])
y_deep = tf.nn.relu(y_deep)

"""final layer"""
# if dfm_params['use_fm'] and dfm_params['use_deep']:
concat_input = tf.concat([fm_first_order, fm_second_order, y_deep], axis=1)
# print(fm_first_order.shape) # N x 37
# print(fm_second_order.shape) # N x 8
# print(y_deep.shape) # N x 32
# print(concat_input.shape) # N x 77
# elif dfm_params['use_fm']:
#     concat_input = tf.concat([fm_first_order, fm_second_order], axis=1)
# elif dfm_params['use_deep']:
#     concat_input = y_deep

out = tf.nn.sigmoid(tf.add(tf.matmul(concat_input,weights['concat_projection']),weights['concat_bias']))
# print(out.shape) # N x 1

"""loss and optimizer"""
loss = tf.losses.log_loss(tf.reshape(label,(-1,1)), out)
optimizer = tf.train.AdamOptimizer(learning_rate=dfm_params['learning_rate'],
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-8).minimize(loss)

"""train"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        epoch_loss,_ = sess.run([loss,optimizer],
                                feed_dict={feat_index:train_feature_index,
                                           feat_value:train_feature_value,
                                           label:train_y}
                                )
        print("epoch %s,loss is %s" % (str(i),str(epoch_loss)))
    predicted = sess.run([out],feed_dict={
        feat_index:test_feature_index,
        feat_value:test_feature_value
    })
    predicted = np.array(predicted).reshape(-1)
    print(predicted)
    for target in predicted:
        if target > 0.5:
            print (target)
# /Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 /Users/wangboyuan/PycharmProjects/ml-100k/MainPipelineDeepFM.py
# /Users/wangboyuan/PycharmProjects/ml-100k/MainPipelineDeepFM.py:23: FutureWarning: Sorting because non-concatenation axis is not aligned. A future version
# of pandas will change to not sort by default.
#
# To accept the future behavior, pass 'sort=False'.
#
# To retain the current behavior and silence the warning, pass 'sort=True'.
#
#   df = pd.concat([dfTrain, dfTest])
# 254
# {'ps_car_01_cat': {10: 0, 11: 1, 7: 2, 6: 3, 9: 4, 5: 5, 4: 6, 8: 7, 3: 8, 0: 9, 2: 10, 1: 11, -1: 12}, 'ps_car_02_cat': {1: 13, 0: 14}, 'ps_car_03_cat': {-1: 15, 0: 16, 1: 17}, 'ps_car_04_cat': {0: 18, 1: 19, 8: 20, 9: 21, 2: 22, 6: 23, 3: 24, 7: 25, 4: 26, 5: 27}, 'ps_car_05_cat': {1: 28, -1: 29, 0: 30}, 'ps_car_06_cat': {4: 31, 11: 32, 14: 33, 13: 34, 6: 35, 15: 36, 3: 37, 0: 38, 1: 39, 10: 40, 12: 41, 9: 42, 17: 43, 7: 44, 8: 45, 5: 46, 2: 47, 16: 48}, 'ps_car_07_cat': {1: 49, -1: 50, 0: 51}, 'ps_car_08_cat': {0: 52, 1: 53}, 'ps_car_09_cat': {0: 54, 2: 55, 3: 56, 1: 57, -1: 58, 4: 59}, 'ps_car_10_cat': {1: 60, 0: 61, 2: 62}, 'ps_car_11': {2: 63, 3: 64, 1: 65, 0: 66}, 'ps_car_11_cat': {12: 67, 19: 68, 60: 69, 104: 70, 82: 71, 99: 72, 30: 73, 68: 74, 20: 75, 36: 76, 101: 77, 103: 78, 41: 79, 59: 80, 43: 81, 64: 82, 29: 83, 95: 84, 24: 85, 5: 86, 28: 87, 87: 88, 66: 89, 10: 90, 26: 91, 54: 92, 32: 93, 38: 94, 83: 95, 89: 96, 49: 97, 93: 98, 1: 99, 22: 100, 85: 101, 78: 102, 31: 103, 34: 104, 7: 105, 8: 106, 3: 107, 46: 108, 27: 109, 25: 110, 61: 111, 16: 112, 69: 113, 40: 114, 76: 115, 39: 116, 88: 117, 42: 118, 75: 119, 91: 120, 23: 121, 2: 122, 71: 123, 90: 124, 80: 125, 44: 126, 92: 127, 72: 128, 96: 129, 86: 130, 62: 131, 33: 132, 67: 133, 73: 134, 77: 135, 18: 136, 21: 137, 74: 138, 37: 139, 48: 140, 70: 141, 13: 142, 15: 143, 102: 144, 53: 145, 65: 146, 100: 147, 51: 148, 79: 149, 52: 150, 63: 151, 94: 152, 6: 153, 57: 154, 35: 155, 98: 156, 56: 157, 97: 158, 55: 159, 84: 160, 50: 161, 4: 162, 58: 163, 9: 164, 17: 165, 11: 166, 45: 167, 14: 168, 81: 169, 47: 170}, 'ps_car_12': 171, 'ps_car_13': 172, 'ps_car_14': 173, 'ps_car_15': 174, 'ps_ind_01': {2: 175, 1: 176, 5: 177, 0: 178, 4: 179, 3: 180, 6: 181, 7: 182}, 'ps_ind_02_cat': {2: 183, 1: 184, 4: 185, 3: 186, -1: 187}, 'ps_ind_03': {5: 188, 7: 189, 9: 190, 2: 191, 0: 192, 4: 193, 3: 194, 1: 195, 11: 196, 6: 197, 8: 198, 10: 199}, 'ps_ind_04_cat': {1: 200, 0: 201, -1: 202}, 'ps_ind_05_cat': {0: 203, 1: 204, 4: 205, 3: 206, 6: 207, 5: 208, -1: 209, 2: 210}, 'ps_ind_06_bin': {0: 211, 1: 212}, 'ps_ind_07_bin': {1: 213, 0: 214}, 'ps_ind_08_bin': {0: 215, 1: 216}, 'ps_ind_09_bin': {0: 217, 1: 218}, 'ps_ind_10_bin': {0: 219, 1: 220}, 'ps_ind_11_bin': {0: 221, 1: 222}, 'ps_ind_12_bin': {0: 223, 1: 224}, 'ps_ind_13_bin': {0: 225, 1: 226}, 'ps_ind_14': {0: 227, 1: 228, 2: 229, 3: 230}, 'ps_ind_15': {11: 231, 3: 232, 12: 233, 8: 234, 9: 235, 6: 236, 13: 237, 4: 238, 10: 239, 5: 240, 7: 241, 2: 242, 0: 243, 1: 244}, 'ps_ind_16_bin': {0: 245, 1: 246}, 'ps_ind_17_bin': {1: 247, 0: 248}, 'ps_ind_18_bin': {0: 249, 1: 250}, 'ps_reg_01': 251, 'ps_reg_02': 252, 'ps_reg_03': 253}
# train_feature_index = (10000, 37)
# train_feature_value = (10000, 37)
# Index(['id', 'ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat',
#        'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin',
#        'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
#        'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin',
#        'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03',
#        'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat',
#        'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat',
#        'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11',
#        'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01',
#        'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
#        'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11',
#        'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
#        'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin',
#        'ps_calc_20_bin'],
#       dtype='object')
# 37
# WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Colocations handled automatically by placer.
# (?, 37)
# (?, 8)
# (?, 32)
# (?, 77)
# WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/python/ops/losses/losses_impl.py:514: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# WARNING:tensorflow:From /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# 2019-06-24 22:09:03.545712: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# epoch 0,loss is 0.7596815
# epoch 1,loss is 0.73606247
# epoch 2,loss is 0.7129119
# epoch 3,loss is 0.69017696
# epoch 4,loss is 0.6678198
# epoch 5,loss is 0.6458216
# epoch 6,loss is 0.62417257
# epoch 7,loss is 0.6028617
# epoch 8,loss is 0.5819006
# epoch 9,loss is 0.5613434
# epoch 10,loss is 0.5412136
# epoch 11,loss is 0.52142215
# epoch 12,loss is 0.50187665
# epoch 13,loss is 0.48256886
# epoch 14,loss is 0.46355024
# epoch 15,loss is 0.4448912
# epoch 16,loss is 0.42671362
# epoch 17,loss is 0.40909365
# epoch 18,loss is 0.39198107
# epoch 19,loss is 0.37532705
# epoch 20,loss is 0.35918394
# epoch 21,loss is 0.34371656
# epoch 22,loss is 0.32923508
# epoch 23,loss is 0.31546783
# epoch 24,loss is 0.30223164
# epoch 25,loss is 0.2897611
# epoch 26,loss is 0.27818593
# epoch 27,loss is 0.26776046
# epoch 28,loss is 0.25821385
# epoch 29,loss is 0.24921171
# epoch 30,loss is 0.24074166
# epoch 31,loss is 0.23280029
# epoch 32,loss is 0.22537959
# epoch 33,loss is 0.21846147
# epoch 34,loss is 0.21202037
# epoch 35,loss is 0.20602827
# epoch 36,loss is 0.20045066
# epoch 37,loss is 0.19529694
# epoch 38,loss is 0.1905777
# epoch 39,loss is 0.18629508
# epoch 40,loss is 0.18244536
# epoch 41,loss is 0.17900243
# epoch 42,loss is 0.1758885
# epoch 43,loss is 0.1731284
# epoch 44,loss is 0.17073438
# epoch 45,loss is 0.1686926
# epoch 46,loss is 0.16698003
# epoch 47,loss is 0.16556536
# epoch 48,loss is 0.16441604
# epoch 49,loss is 0.16351415
# epoch 50,loss is 0.16284122
# epoch 51,loss is 0.1623702
# epoch 52,loss is 0.16207227
# epoch 53,loss is 0.16191757
# epoch 54,loss is 0.16187686
# epoch 55,loss is 0.16192119
# epoch 56,loss is 0.16202247
# epoch 57,loss is 0.16215472
# epoch 58,loss is 0.1622956
# epoch 59,loss is 0.16242741
# epoch 60,loss is 0.16253747
# epoch 61,loss is 0.16261743
# epoch 62,loss is 0.1626629
# epoch 63,loss is 0.16267261
# epoch 64,loss is 0.16264749
# epoch 65,loss is 0.1625905
# epoch 66,loss is 0.16250558
# epoch 67,loss is 0.16239746
# epoch 68,loss is 0.16227107
# epoch 69,loss is 0.16213152
# epoch 70,loss is 0.16199051
# epoch 71,loss is 0.16186015
# epoch 72,loss is 0.16173178
# epoch 73,loss is 0.16160162
# epoch 74,loss is 0.16147184
# epoch 75,loss is 0.16134419
# epoch 76,loss is 0.16122174
# epoch 77,loss is 0.16110557
# epoch 78,loss is 0.16099651
# epoch 79,loss is 0.16089492
# epoch 80,loss is 0.16080093
# epoch 81,loss is 0.16071399
# epoch 82,loss is 0.16063343
# epoch 83,loss is 0.16055922
# epoch 84,loss is 0.1604917
# epoch 85,loss is 0.16042888
# epoch 86,loss is 0.16036998
# epoch 87,loss is 0.16031435
# epoch 88,loss is 0.16026111
# epoch 89,loss is 0.16020964
# epoch 90,loss is 0.16015919
# epoch 91,loss is 0.1601092
# epoch 92,loss is 0.16005908
# epoch 93,loss is 0.16000845
# epoch 94,loss is 0.15995693
# epoch 95,loss is 0.15990432
# epoch 96,loss is 0.15985046
# epoch 97,loss is 0.15979546
# epoch 98,loss is 0.15973923
# epoch 99,loss is 0.1596819