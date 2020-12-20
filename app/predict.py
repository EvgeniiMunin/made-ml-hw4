from keras.layers import Activation, Dense, Dropout, LSTM
from keras.models import Sequential


def buildLstmModel(
    X,
    incur,
    outcur,
    output_size=1,
    neurons=100,
    activ_func="linear",
    dropout=0.2,
    loss="mse",
    optimizer="adam",
):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    model.load_weights(
        './made-ml-hw4/app/saved_models/model_v1_lstm800_mod_win_7d_BTC_USD.h5'.format(incur, outcur)
#         "./saved_models/model_v1_lstm800_mod_win_7d_{}_{}.h5".format(incur, outcur)
    )
    return model


def predict(model, X):
    return model.predict(X).squeeze()
