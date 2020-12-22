import pandas as pd
from app import read_preprocess
from app import predict
import matplotlib.pyplot as plt

def get_predict():

    DAYS_BACK = 200
    PRED_HORIZON = 7
    WIN_LEN = 30
    INCUR = "BTC"
    OUTCUR = "USD"
    
    FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
    TARGET_COL = "close"

#if __name__ == "__main__":
    df = read_preprocess.parseData(DAYS_BACK, INCUR, OUTCUR) # ok
    df, X, y_test = read_preprocess.prepare_data(
        df, TARGET_COL, window_len=WIN_LEN, pred_horizon=PRED_HORIZON
    )
    print(df)

    model = predict.buildLstmModel(X, INCUR, OUTCUR)
    preds = predict.predict(model, X)
    targets = df[TARGET_COL][WIN_LEN+PRED_HORIZON:]

    # create future date column
    pred_dates = []
    for offset in range(1, PRED_HORIZON + 1):
        pred_dates.append(targets.index[-1] + pd.DateOffset(offset))
    pred_index = targets.index[1:].append(pd.Index(pred_dates))
    print(pred_index.shape, pred_index)

    # denormalize historic preds on its end of window target value
    hist_preds = targets.values[:-1] * (preds[1:-PRED_HORIZON] + 1)

    # denormalize future preds check
    preds_denorm = []
    for i in range(preds[-PRED_HORIZON:].shape[0]):
        if i == 0:
            prev_val = targets.values[-1] * (preds[i]+ 1)
        else:
            prev_val = prev_val * (preds[i]+ 1)
        preds_denorm.append(prev_val)
 
    print(hist_preds.shape, len(preds_denorm), targets.shape, preds.shape)
    temp = list(hist_preds) + preds_denorm
    print(pred_index.shape, len(temp))
    out_preds = pd.Series(index=pred_index, data=temp)
    preds_denorm = pd.Series(preds_denorm, index=pred_dates)

    return out_preds, targets, preds_denorm, df

#    # plot preds and targets
#    plt.figure(figsize=(10,8))
#    out_preds.plot()
#    targets.plot()
#    plt.grid()
#    plt.legend(['prediction', 'past real'])
#    plt.show()





