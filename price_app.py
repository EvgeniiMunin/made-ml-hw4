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



plt.figure(figsize=(5,25))
# plt.subplots_adjust(top = 1, bottom = 0)
mean = 10; std = 2
y = np.random.randn(10).reshape(-1,1) * std + mean
df = pd.DataFrame(y, columns=['BTC'])
df.plot(figsize=(10, 4))
st.pyplot(plt)

# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="crypto.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df), unsafe_allow_html=True)


# with st.spinner(text='In progress'):
#     time.sleep(2)
#     st.success('Done')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
# st.balloons()









