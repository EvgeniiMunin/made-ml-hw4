import pandas as pd
from app import read_preprocess
from app import predict
import matplotlib.pyplot as plt

DAYS_BACK = 200
WIN_LEN = 30

ALL_FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
TARGET_COL = "close"


def get_predict(df, incur, outcur, pred_horizon):
    df, X, y_test = read_preprocess.prepare_data(
        df, TARGET_COL, window_len=WIN_LEN, pred_horizon=pred_horizon
    )
    print(df)

    model = predict.buildLstmModel(X, incur, outcur)
    preds = predict.predict(model, X)
    targets = df[TARGET_COL][WIN_LEN + pred_horizon :]

    # create future date column
    pred_dates = []
    for offset in range(1, pred_horizon + 1):
        pred_dates.append(targets.index[-1] + pd.DateOffset(offset))
    pred_index = targets.index[1:].append(pd.Index(pred_dates))
    print(pred_index.shape, pred_index)

    # denormalize historic preds on its end of window target value
    hist_preds = targets.values[:-1] * (preds[1:-pred_horizon] + 1)

    # denormalize future preds check
    preds_denorm = []
    for i in range(preds[-pred_horizon:].shape[0]):
        prev_val = targets.values[-1] * (preds[i] + 1)
        preds_denorm.append(prev_val)

    print(hist_preds.shape, len(preds_denorm), targets.shape, preds.shape)
    temp = list(hist_preds) + preds_denorm
    print(pred_index.shape, len(temp))
    out_preds = pd.Series(index=pred_index, data=temp)
    preds_denorm = pd.Series(preds_denorm, index=pred_dates)

    return out_preds, targets, preds_denorm, df
