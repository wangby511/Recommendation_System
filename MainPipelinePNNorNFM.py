import pandas as pd
from PNN import PNN
from NFM import NFM
from DeepCrossNetwork import DCN

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

def load_data():
    dfTrain = pd.read_csv(TRAIN_FILE) # shape (10000, 59)
    dfTest = pd.read_csv(TEST_FILE) # shape (2000, 58)
    df = pd.concat([dfTrain, dfTest], sort=True)
    # print (df.shape) # shape (12000, 59)

    cols = [c for c in dfTrain.columns if c not in ['id', 'target']]
    cols = [c for c in cols if (c not in IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['id'].values

    return df, dfTrain, dfTest, X_train, y_train, X_test, ids_test

def split_dimensions(df):
    feat_dict = {}
    tc = 0
    for col in df.columns:
        if col in IGNORE_COLS:
            continue

        if col in NUMERIC_COLS:
            feat_dict[col] = tc
            tc += 1

        else:
            us = df[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
            tc += len(us)
    feat_dimension = tc
    # print (feat_dict)
    # print (feat_dimension) # 254
    return feat_dict,feat_dimension

def data_parse(df_data, feat_dict, training = True):

    if training:
        y = df_data['target'].values.tolist()
        df_data.drop(['id', 'target'], axis=1, inplace=True)
    else:
        ids = df_data['id'].values.tolist()
        df_data.drop(['id'], axis=1, inplace=True)

    df_index = df_data.copy()
    for col in df_data.columns:
        if col in IGNORE_COLS:
            df_data.drop(col, axis = 1, inplace = True)
            df_index.drop(col, axis = 1, inplace = True)
            continue
        if col in NUMERIC_COLS:
            df_index[col] = feat_dict[col]
        else:
            df_index[col] = df_data[col].map(feat_dict[col])
            df_data[col] = 1.

    xi = df_index.values.tolist()
    xd = df_data.values.tolist()

    if training:
        return xi, xd, y
    else:
        return xi, xd, ids


def run_base_model_pnn(fd, dfTrain):
    split_dimensions(fd)



def main():
    fd, dfTrain, dfTest, X_train, y_train, X_test, ids_test  = load_data()

    feat_dict, feat_dimension = split_dimensions(fd)

    Xi_train, Xv_train, y_train = data_parse(dfTrain, feat_dict, training=True)

    Xi_test, Xv_test, ids_test = data_parse(dfTest, feat_dict, training=False)

    # print(dfTrain.dtypes)

    feature_size = feat_dimension
    field_size = len(Xi_train[0])

    print(feature_size,field_size) # 254, 37

    # pnn_model = PNN(feature_size = feat_dimension,
    #                 field_size = len(Xi_train[0]),
    #                 batch_size=128,
    #                 epoch=100
    #                 )
    #
    # pnn_model.fit(Xi_train,Xv_train,y_train)

    # nfm_model = NFM(feature_size = feat_dimension,
    #                 field_size = len(Xi_train[0]),
    #                 batch_size=128,
    #                 epoch=5
    #                 )
    # nfm_model.fit(Xi_train,Xv_train,y_train)

    dcn_model = DCN(feature_size=feat_dimension,
                    field_size=len(Xi_train[0]),
                    batch_size=128,
                    epoch=1
                    )
    dcn_model.fit(Xi_train, Xv_train, y_train)



if __name__ == '__main__':
    main()