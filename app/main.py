import pandas as pd
import json
import requests
import numpy as np
import read_preprocess
import predict

DAYS_BACK = 200
PRED_HORIZON = 7
WIN_LEN = 30
INCUR = 'BTC'
OUTCUR = 'USD'

FEATURES = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
TARGET_COL = 'close'

if __name__ == '__main__':
    df = read_preprocess.parseData(DAYS_BACK, INCUR, OUTCUR)
    df, X, y_test, dates = read_preprocess.prepare_data(df, TARGET_COL,
                                                        window_len=WIN_LEN,
                                                        pred_horizon=PRED_HORIZON)

    model = predict.buildLstmModel(X, INCUR, OUTCUR)
    preds = predict.predict(model, X)
    targets = df[TARGET_COL][WIN_LEN:]
    # preds = preds[PRED_HORIZON:]
    print(targets)
    print(preds.shape, y_test.shape)

    # create future date column
    pred_dates = []
    for offset in range(1, PRED_HORIZON + 1):
        pred_dates.append(targets.index[-1] + pd.DateOffset(offset))
    pred_index = targets.index[PRED_HORIZON:].append(pd.Index(pred_dates))
    print(pred_index.shape, pred_index)

    # denormalize future predictions
    preds_denorm = []
    prev_pred = targets[-1]
    for i in range(targets[-PRED_HORIZON:].shape[0]):
        pred_val = targets[-PRED_HORIZON:][i] * (preds[i] + 1)
        print(targets[-PRED_HORIZON:][i], preds[i], pred_val)
        preds_denorm.append(pred_val)

    out_preds = targets * (preds + 1)
    print(out_preds.shape, len(preds_denorm))
    temp = list(out_preds[PRED_HORIZON:]) + preds_denorm
    out_preds = pd.Series(index=pred_index, data=temp)
    print(out_preds)
