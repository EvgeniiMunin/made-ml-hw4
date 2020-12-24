import pandas as pd
import pickle


LAG_START = 1
LAG_END = 40


def load_model(file_name):
    """
    Load model from file
    :param file_name: String
        Path to the model file
    :return: predictive model
    """
    with open(file_name, "rb") as file:
        return pickle.load(file)


def get_predict(dforig, incur, outcur, pred_horizon):
    df = dforig.copy(deep=True)
    # add lags of the initial series as features
    for i in range(LAG_START, LAG_END):
        df["lag_{}".format(i)] = df.close.shift(i)
    df.dropna(inplace=True)

    dftest = pd.DataFrame()
    df.reset_index(inplace=True)
    # features until the current date
    for i, row in df[LAG_END : len(df)].iterrows():
        # add features
        for col in df.columns:
            dftest.loc[i, col] = row[col]

    # get x, ytest
    timestamps = dftest["time"]
    xtest = dftest.drop(["time", "volumefrom", "volumeto",], axis=1,).reset_index(
        drop=True
    )

    # load model + predict
    model = load_model("app/saved_models/linreg_{}_{}.pkl".format(incur, outcur))
    preds = pd.DataFrame(
        model.predict(xtest),
        columns=["close1", "close2", "close3", "close4", "close5", "close6", "close7"],
    )
    preds.index = timestamps
    out_preds = preds["close1"][:-1]
    new_preds = preds.iloc[len(preds) - 1]
    dftest.index = timestamps

    print(pred_horizon)
    print(out_preds)
    print(new_preds)
    print(df[LAG_END:])

    # create future date column
    pred_dates = []
    for offset in range(1, pred_horizon + 1):
        pred_dates.append(preds.index[-1] + pd.DateOffset(offset))
    pred_index = pd.Index(pred_dates)
    new_preds = pd.Series(new_preds.values[:pred_horizon])
    new_preds.index = pred_index

    return out_preds, new_preds
