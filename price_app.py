import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import datetime
import base64
import time
import os
import pickle

register_matplotlib_converters()
plt.style.use('default')

import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, './made-ml-hw4/app')
from app import read_preprocess
from app import predict

st.set_page_config(layout="wide")

st.title('Crypto Price App')
st.markdown('''
This app retrieves cryptocurrency prices for the top 100 cryptocurrency from the **CoinMarketCap**!
''')

expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn, BeautifulSoup, requests, json, time
* **Data source:** [CoinMarketCap](http://coinmarketcap.com).
* **Credit:** Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf).
""")

col1 = st.sidebar
# col2, col3 = st.beta_columns((2,1))

col1.header('Input Options')
currency_price_unit = col1.selectbox('Select currency for predict', ('USD', 'BTC', 'ETH'))
predict_period = col1.slider('Predict period', min_value=1, max_value=7)
predict_from = col1.date_input('Predict from')


DAYS_BACK = 200
PRED_HORIZON = 7
WIN_LEN = 30
INCUR = "BTC"
OUTCUR = "USD"

FEATURES = ["high", "low", "open", "volumefrom", "volumeto", "close"]
TARGET_COL = "close"

@st.cache
def data_preprocess():
    # Mutate bar
    df = read_preprocess.parseData(DAYS_BACK, INCUR, OUTCUR)
#     df, X, y_test, dates = read_preprocess.prepare_data(
#         df, TARGET_COL, window_len=WIN_LEN, pred_horizon=PRED_HORIZON
#     )
    return read_preprocess.prepare_data(
        df, TARGET_COL, window_len=WIN_LEN, pred_horizon=PRED_HORIZON
    )

df, X, y_test, dates = data_preprocess()
# st.dataframe(df)

# @st.cache
# def build_model():
#     model = predict.buildLstmModel(X, INCUR, OUTCUR)
#     return model
# model = build_model()

model = predict.buildLstmModel(X, INCUR, OUTCUR)
preds = predict.predict(model, X)
targets = df[TARGET_COL][WIN_LEN:]
preds = preds[PRED_HORIZON:]
st.dataframe(preds)

# create future date column
pred_dates = []
for offset in range(1, PRED_HORIZON + 1):
    pred_dates.append(targets.index[-1] + pd.DateOffset(offset))
pred_index = targets.index[PRED_HORIZON:].append(pd.Index(pred_dates))

# denormalize future preds
preds_denorm = []
prev_pred = targets[-1]
for i in range(targets[-PRED_HORIZON:].shape[0]):
    pred_val = targets[-PRED_HORIZON:][i] * (preds[-PRED_HORIZON + i] + 1)
    # print(targets[-PRED_HORIZON:][i], preds[i], pred_val)
    preds_denorm.append(pred_val)
    
# denormalize historic preds on its end of window target value
out_preds = targets[:-PRED_HORIZON] * (preds + 1)
# print(out_preds.shape, len(preds_denorm), targets.shape, preds.shape)
temp = list(out_preds) + preds_denorm
# print(pred_index.shape, len(temp))
out_preds = pd.Series(index=pred_index, data=temp)

# st.dataframe(targets)
ax = targets.plot(figsize=(10, 5), label='past')
ax.axvline(x=targets.index[-1], color='silver', label='dividing line')
preds_denorm = pd.Series(preds_denorm, index=pred_dates)
# st.dataframe(preds_denorm)
preds_denorm.plot(ax=ax, marker='.', label='prediction')
out_preds[:-PRED_HORIZON].plot(ax=ax, marker='.', label='prediction past')

# denormalize historic preds on its end of window target value
# out_preds = targets[:-PRED_HORIZON] * (preds + 1)
# temp = list(out_preds) + preds_denorm
# out_preds = pd.Series(index=pred_index, data=temp, name=f'{INCUR} / {OUTCUR}')
# st.dataframe(out_preds)


# plt.figure(figsize=(5,25))
# mean = 10; std = 2
# y = np.random.randn(10).reshape(-1,1) * std + mean
# df = pd.DataFrame(y, columns=['BTC'])
# df.plot(figsize=(10, 4))

# ax = out_preds.plot(figsize=(10, 4))
ax.set(ylabel=f'{OUTCUR}', title=f'Prediction of the course {INCUR} / {OUTCUR}')
plt.legend()
# plt.box(False)
# plt.grid()
st.pyplot(plt)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a style="text-align: center" href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)







