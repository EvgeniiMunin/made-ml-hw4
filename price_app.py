import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
from pandas.plotting import register_matplotlib_converters
from datetime import datetime, timedelta
import base64
from app import read_preprocess, predict_lstm, predict_linreg

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import mean_absolute_error

register_matplotlib_converters()
plt.style.use("default")

# constants
DAYS_BACK = 200
WIN_LEN = 30
ALL_FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
TARGET_COL = "close"

st.set_page_config(layout="wide")

st.title("Currency Rate Prediction App")
st.markdown(
    """
This app retrieves currency prices for Bitcoin, USD, EUR, RUB from **min-api crypto compare** 
and predicts the close exchange rate for several days in the future.
"""
)

expander_bar = st.beta_expander("About")
expander_bar.write(
    """
* **Team:** Andrei Starikov, Nikolai Diakin, Ilya Avilov, Orkhan Gadzhily, Evgenii Munin
* **Python libraries:** scikit-learn, keras, base64, streamlit, plotly, pandas, numpy, requests, json
* **Data source:** Data resource API is available at [min-api crypto compare](https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=CAD&limit=500)
"""
)

# define left column sidebar
col1 = st.sidebar

# add logo
st.sidebar.image(image=Image.open("logo.jpeg"), width=200)

# select currency
col1.header("Input Options")
currency_price_unit = col1.selectbox(
    "Select currency for predict",
    (
        "BTC EUR",
        "BTC RUB",
        "BTC USD",
        "EUR BTC",
        "EUR USD",
        "RUB BTC",
        "USD BTC",
        "USD EUR",
    ),
)
incur = currency_price_unit.split()[0]
outcur = currency_price_unit.split()[1]
print("check price unit: ", currency_price_unit, incur, outcur)

# select period
pred_horizon = col1.slider("Predict period (days)", min_value=1, max_value=3)
print("check pred horizon: ", pred_horizon)

# select calendar date
predict_from = col1.date_input("Predict from")
predict_from += timedelta(days=1)
print("check predict from: ", predict_from, type(predict_from))

# select model
model_choice = col1.selectbox("Select model", ("LSTM (slide win)", "Lin Reg (lags 40)"))

# check target dates with pred_from+pred_horizon
time_back = DAYS_BACK + (datetime.now().date() - predict_from).days
df = read_preprocess.parseData(time_back, incur, outcur, ALL_FEATURES)
dforig = read_preprocess.parseData(time_back, incur, outcur, ALL_FEATURES)
df = df[df.index < datetime(predict_from.year, predict_from.month, predict_from.day)]

# compute preds
if model_choice == "LSTM (slide win)":
    preds, _, new_preds = predict_lstm.get_predict(df, incur, outcur, pred_horizon)
else:
    preds, new_preds = predict_linreg.get_predict(df, incur, outcur, pred_horizon)
print("check")
print(preds)
print(preds.to_frame())

dfhist = dforig[
    (dforig.index >= preds.index.min()) & (dforig.index <= new_preds.index.max())
]
preds = preds.to_frame()
preds.columns = ["hist preds"]

# plots. fill in bewteen max min
fig = px.line(preds)
fig.add_trace(go.Scatter(x=dfhist.index, y=dfhist["close"], mode="lines", name="hist"))
fig.add_trace(
    go.Scatter(
        x=dfhist.index,
        y=dfhist["high"],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
        showlegend=False,
        name="upper bound",
    )
)
fig.add_trace(
    go.Scatter(
        x=dfhist.index,
        y=dfhist["low"],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
        showlegend=False,
        name="lower bound",
    )
)

fig.add_trace(
    go.Scatter(x=new_preds.index, y=new_preds, mode="markers", name="new preds")
)

st.plotly_chart(fig, use_container_width=True)

# Download CSV data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a style="text-align: center" href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href


# show metrics
mae = mean_absolute_error(dfhist.iloc[-5:]["close"], preds[-6:-1])
if mae > 0.01:
    st.markdown("Last 5 days MAE: {}".format(round(mae, 2)))
else:
    st.markdown("Last 5 days MAE: {}".format(mae))

st.header("Prediction")
st.dataframe(new_preds)

st.markdown(filedownload(df), unsafe_allow_html=True)
