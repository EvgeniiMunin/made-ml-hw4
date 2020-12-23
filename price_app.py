import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from datetime import datetime
import base64
from app import read_preprocess, main

register_matplotlib_converters()
plt.style.use("default")

# constants
DAYS_BACK = 200
WIN_LEN = 30
ALL_FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
TARGET_COL = "close"

st.set_page_config(layout="wide")

st.title("Crypto Price App")
st.markdown(
    """
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!
"""
)

expander_bar = st.beta_expander("About")
expander_bar.markdown(
    """
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [CoinMarketCap](http://coinmarketcap.com).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
"""
)

# define left column sidebar
col1 = st.sidebar

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
pred_horizon = col1.slider("Predict period (days)", min_value=1, max_value=7)
print("check pred horizon: ", pred_horizon)

# select calendar date
predict_from = col1.date_input("Predict from")
print("check predict from: ", predict_from, type(predict_from))

# check target dates with pred_from+pred_horizon
time_back = DAYS_BACK + (datetime.now().date() - predict_from).days
df = read_preprocess.parseData(time_back, incur, outcur, ALL_FEATURES)
df = df[df.index < datetime(predict_from.year, predict_from.month, predict_from.day)]

# compute preds
out_preds, targets, preds_denorm, df = main.get_predict(df, incur, outcur, pred_horizon)

# plots
ax = targets.plot(figsize=(10, 5), label="past")
ax.axvline(x=targets.index[-1], color="silver", label="dividing line")
preds_denorm.plot(ax=ax, marker=".", label="prediction")
out_preds[:-pred_horizon].plot(ax=ax, marker=".", label="prediction past")

ax.set(ylabel=f"{outcur}", title=f"Prediction of the course {incur} / {outcur}")
plt.legend()
plt.box(False)
plt.grid()
st.pyplot(plt)

# Download CSV data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a style="text-align: center" href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df), unsafe_allow_html=True)
