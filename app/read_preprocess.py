import pandas as pd
import numpy as np
import requests
import json

FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
TARGET_COL = "close"


def parseData(days_back, incur, outcur, features):
    endpoint = "https://min-api.cryptocompare.com/data/histoday"
    res = requests.get(
        endpoint + "?fsym={}&tsym={}&limit={}".format(incur, outcur, days_back)
    )
    df = pd.DataFrame(json.loads(res.content)["Data"])
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index, unit="s")
    return df[features]


def extract_window_data(df, window_len=5, zero_base=True):
    window_data, dates = [], []

    # extract window_len points for win segment. get only features
    copydf = df.copy(deep=True).drop(["close"], axis=1)

    for idx in range(len(copydf) - window_len):
        tmp = copydf[idx : (idx + window_len)].copy(deep=True)

        if zero_base:
            tmp = normalise_zero_base(tmp)

        dates.append(tmp.index)
        window_data.append(tmp.values)
        del tmp

    del copydf
    return np.array(window_data), dates


def prepare_data(df, target_col, window_len=10, pred_horizon=5, zero_base=True):
    # process data get tensors by win
    X, dates = extract_window_data(df, window_len, zero_base)

    # extract targets - the point after each win_len segment + offset of predict_horizon
    y_test = df[target_col][window_len + pred_horizon :]
    y_test_prevs = df[target_col][window_len + pred_horizon - 1 : -1]

    # normalize y on its end of window price
    y_test = y_test.values / y_test_prevs.values - 1

    #    y_test = y_test.values
    #    y_test = y_test / df[target_col][window_len:][:-pred_horizon].values - 1
    #    y_test = y_test / df[target_col][:-window_len].values - 1

    return df, X, y_test


def normalise_zero_base(df):
    return df / df.iloc[0] - 1
