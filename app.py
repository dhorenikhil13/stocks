# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 18:02:40 2025

@author: dhore
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ---- Functions ----
def get_spy_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    tickers = [ticker.replace('.', '-') for ticker in df['Symbol']]
    return tickers

def get_price_and_volume(tickers, date):
    data = {}
    start_date = date
    end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if not df.empty:
            price = df['Close'].iloc[0].item()
            volume = df['Volume'].iloc[0].item()
            data[ticker] = {
                'Price': f"{price:.2f}",
                'Volume': f"{int(volume):,}"
            }
        else:
            data[ticker] = {'Price': 'N/A', 'Volume': 'N/A'}
    return pd.DataFrame.from_dict(data, orient='index')

# ---- Streamlit UI ----
st.title("S&P 500 Stock Price Viewer")
date_input = st.date_input("Pick a date", value=datetime.today() - timedelta(days=1))
num_tickers = st.slider("Number of tickers", 5, 10, 50, 100, 250, 500)

if st.button("Fetch Data"):
    tickers = get_spy_tickers()[:num_tickers]
    df = get_price_and_volume(tickers, date_input.strftime("%Y-%m-%d"))
    st.dataframe(df)
    st.download_button("Download as CSV", df.to_csv().encode('utf-8'), "stock_data.csv", "text/csv")

