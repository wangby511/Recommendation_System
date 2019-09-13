import sys
import pandas as pd
import numpy as np
from scipy.sparse import csr
from FM import FM

# reference: https://github.com/babakx/fm_tensorflow/blob/master/fm_tensorflow.ipynb

def parseData(train_users_values, train_items_values, test_users_values, test_items_values):
    """

    :param train_users_values: userID for training data
    :param train_items_values: itemID for training data
    :param test_users_values: userID for testing data
    :param test_items_values: itemID for testing data
    :return: sparse matrix only contains '1's and '0's for both training data and test data
    each row only contain two '1's in the positions of user index and item index
    n x p
    n = size of total data
    p = number of users + number of items
    in this example, n = 90570, p = 943 + 1680 = 2623
    """

    # calculate users both
    users_index = {}

    count = 0
    for user in train_users_values:
        if user not in users_index:
            users_index[user] = count
            count = count + 1

    for user in test_users_values:
        if user not in users_index:
            users_index[user] = count
            count = count + 1

    # calculate items both
    items_index = {}

    count = 0
    for item in train_items_values:
        if item not in items_index:
            items_index[item] = count
            count = count + 1

    for item in test_items_values:
        if item not in items_index:
            items_index[item] = count
            count = count + 1

    n_train = len(list(train_users_values))  # num samples
    n_test = len(list(test_users_values))

    N_user = len(users_index)  # user number

    N_item = len(items_index)  # item number

    p = N_user + N_item

    print("n_train =",n_train,", n_test =",n_test,", N_user =",N_user,", N_item =",N_item,",p =",p)
    # n_train = 90570, n_test = 9430, N_user = 943, N_item = 1682, p = 2625

    # build sparse matrix for training data
    train_data = np.ones((2 * n_train, ))

    train_indptr = np.zeros((n_train  +  1 , ))
    for i in range(1, n_train + 1):
        train_indptr[i] = train_indptr[i - 1] + 2

    index = 0
    train_indices = np.zeros((2 * n_train, ))
    for i in range(n_train):
        train_indices[index] = users_index[train_users_values[i]]
        index = index + 1
        train_indices[index] = N_user + items_index[train_items_values[i]]
        index = index + 1

    # build sparse matrix for test data
    test_data = np.ones((2 * n_test,))

    test_indptr = np.zeros((n_test + 1,))
    for i in range(1, n_test + 1):
        test_indptr[i] = test_indptr[i - 1] + 2

    index = 0
    test_indices = np.zeros((2 * n_test,))
    for i in range(n_test):
        test_indices[index] = users_index[test_users_values[i]]
        index = index + 1
        test_indices[index] = N_user + items_index[test_items_values[i]]
        index = index + 1

    return csr.csr_matrix((train_data, train_indices, train_indptr), shape=(n_train, p)),\
           csr.csr_matrix((test_data, test_indices, test_indptr), shape=(n_test, p))

def main():
    cols = ['user', 'item', 'rating', 'timestamp']
    train = pd.read_csv('ml-100k/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('ml-100k/ua.test', delimiter='\t', names=cols)

    y_train = train.rating.values
    y_test= test.rating.values

    # vectorize data and convert them to csr matrix
    X_train, X_test = parseData(train.user.values, train.item.values, test.user.values, test.item.values)

    X_train = X_train.todense()
    X_test = X_test.todense()

    # print shape of data
    print(X_train.shape) #(90570, 2625)
    print(X_test.shape) #(9430, 2625)

    # number of latent factors
    k = 10
    n, p = X_train.shape
    print("n =",n,", p= ",p)
    # n = 90570, p = 2625

    fm = FM(feature_number=p,embedding_size=k)
    fm.fit(X_train,y_train,X_test,y_test)

if __name__ == '__main__':
    main()

# 2019 - 09 - 1216: 04:29.001332: Itensorflow / core / platform / cpu_feature_guard.cc: 141] Your CPU supports instructions
# that this TensorFlow binary was not compiled to use: AVX2 FMA
# epoch = 0 , RMSE =  3.2126021
# epoch = 1 , RMSE =  2.7656739
# epoch = 2 , RMSE =  2.4024153
# epoch = 3 , RMSE =  2.1098776
# epoch = 4 , RMSE =  1.8768882
# epoch = 5 , RMSE =  1.6934283
# epoch = 6 , RMSE =  1.5508204
# epoch = 7 , RMSE =  1.4416351
# epoch = 8 , RMSE =  1.3588146
# epoch = 9 , RMSE =  1.2967539
# epoch = 10 , RMSE =  1.250724
# epoch = 11 , RMSE =  1.2167224
# epoch = 12 , RMSE =  1.1917113
# epoch = 13 , RMSE =  1.173357
# epoch = 14 , RMSE =  1.1599034
# epoch = 15 , RMSE =  1.1499834
# epoch = 16 , RMSE =  1.1426805
# epoch = 17 , RMSE =  1.1372212
# epoch = 18 , RMSE =  1.1331489
# epoch = 19 , RMSE =  1.1300613
# epoch = 20 , RMSE =  1.1277224
# epoch = 21 , RMSE =  1.125948
# epoch = 22 , RMSE =  1.1245725
# epoch = 23 , RMSE =  1.1234976
# epoch = 24 , RMSE =  1.1226429
# epoch = 25 , RMSE =  1.1219419
# epoch = 26 , RMSE =  1.1213807
# epoch = 27 , RMSE =  1.1209167
# epoch = 28 , RMSE =  1.1205349
# epoch = 29 , RMSE =  1.1202054