import random

import xgboost as xgb
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # read data
    # data = np.random.rand(5, 10)
    # label = np.random.randint(2, size=5)
    data_path = "data/xgb-data.csv"
    data = pd.read_csv(data_path, encoding="utf-8")
    sp = data.shape
    print(type(data))
    print(data)
    train_data = data.iloc[:int(sp[0] * 0.8), :]
    test_data = data.iloc[int(sp[0] * 0.8):, :]
    print(type(train_data))
    print(train_data)
    train_x = train_data.iloc[:, :6]
    train_y = train_data.iloc[:, 6]
    test_x = test_data.iloc[:, :6]
    test_y = test_data.iloc[:, 6]

    print(train_x)
    print(">>>>>")
    print(train_y)
    print("======")
    print(test_x)
    print("-----")
    print(test_y)

    xg_train = xgb.DMatrix(train_x, train_y)
    xg_test = xgb.DMatrix(test_x, test_y)

    # init model
    params = {
        "booster": "gbtree",
        "objective": "multi:softmax",
        "gamma": 0.1,
        "eta": 0.1,
        "nthread": 4,
        "max_depth": 6,
        "num_class": 3,
        "silent": 0,
        "seed": random.randint(1, 100)
    }
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 5
    print(">> Model init done..")
    # train
    bst = xgb.train(params, xg_train, num_round, watchlist)
    print(">> Train model done..")

    # test
    pred = bst.predict(xg_test)
    error_rate = np.sum(pred != test_y) / test_y.shape[0]
    print("precision: {}".format(error_rate))
    bst.save_model("models/xgb/xgb_v1.json")

