# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:57:49 2025

@author: dhore
"""

import yfinance as yf
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from tqdm import tqdm

# === Get current price ===
def get_current_price(ticker):
    try:
        df = yf.download(ticker, period="1d", progress=False)
        if not df.empty:
            return df['Close'].iloc[0].item()
    except:
        return None
    return None

# === Get tickers by index: 'SPY' or 'QQQ' ===
def get_index_tickers(index="SPY"):
    if index == "SPY":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table_index = 0
        column = "Symbol"
    elif index == "QQQ":
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        table_index = 4
        column = "Ticker"
    else:
        raise ValueError("Unsupported index. Choose 'SPY' or 'QQQ'.")

    try:
        tables = pd.read_html(url)
        df = tables[table_index]
        tickers = [t.replace('.', '-') for t in df[column].tolist()]
        return tickers
    except Exception as e:
        st.error(f"❌ Failed to fetch {index} tickers: {e}")
        return []

# === Filter stocks above $100 ===
def filter_high_price_tickers(tickers, threshold=100):
    high_price = []
    for ticker in tqdm(tickers, desc=f"Filtering > ${threshold}"):
        price = get_current_price(ticker)
        if price is not None and isinstance(price, (int, float)) and price > threshold:
            high_price.append(ticker)
    return high_price

# === Calculate % change ===
def get_price_and_volume_change(tickers, start_date, end_date):
    data = []
    for ticker in tqdm(tickers, desc="Fetching % changes"):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False
            )
            if df.empty or len(df) < 2:
                continue
            open_price = df['Close'].iloc[0]
            close_price = df['Close'].iloc[-1]
            open_vol = df['Volume'].iloc[0]
            close_vol = df['Volume'].iloc[-1]

            price_pct_change = ((close_price - open_price) / open_price) * 100
            volume_pct_change = ((close_vol - open_vol) / open_vol) * 100

            data.append({
                'Ticker': ticker,
                'Price Change %': round(price_pct_change, 2),
                'Volume Change %': round(volume_pct_change, 2),
                'Price': round(close_price, 2)
            })
        except Exception as e:
            continue
    return pd.DataFrame(data)

# === Plotly Bubble Chart ===
def plot_interactive_bubble(df, index_name, start_date, end_date):
    def categorize(row):
        p = row['Price Change %']
        v = row['Volume Change %']
        if p < 0 and v > 100:
            return 'Sell - High Vol'
        elif p < 0:
            return 'Sell - Low Vol'
        elif 0 <= p < 30 and v > 100:
            return 'Hold - High Vol'
        elif 0 <= p < 30:
            return 'Hold - Low Vol'
        elif p >= 30 and v > 100:
            return 'Star - High Vol'
        else:
            return 'Star - Low Vol'

    df['Category'] = df.apply(categorize, axis=1)

    color_map = {
        'Sell - High Vol': '#8B0000',
        'Sell - Low Vol': '#FFA07A',
        'Hold - High Vol': '#00008B',
        'Hold - Low Vol': '#ADD8E6',
        'Star - High Vol': '#006400',
        'Star - Low Vol': '#90EE90'
    }

    df['Color'] = df['Category'].map(color_map)

    fig = px.scatter(
        df,
        x='Price Change %',
        y='Volume Change %',
        color='Category',
        color_discrete_map=color_map,
        hover_data={
            'Ticker': True,
            'Price': True,
            'Price Change %': ':.2f',
            'Volume Change %': ':.2f',
            'Category': False,
            'Color': False
        },
        size_max=30,
        size=[10]*len(df),
        opacity=0.7
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        title=f'{index_name} Stocks: Price vs Volume Change ({start_date} → {end_date})',
        xaxis_title='Price % Change',
        yaxis_title='Volume % Change',
        template='plotly_white',
        height=700
    )
    return fig

# === Streamlit App ===
st.set_page_config(layout="wide")
st.title("📊 Stock Bubble Chart: Price vs Volume Change")

index_name = st.selectbox("Choose Index", ["QQQ", "SPY"], index=0)
start_date = st.date_input("Start Date", datetime(2025, 1, 2))
end_date = st.date_input("End Date", datetime(2025, 7, 22))

if st.button("Run Analysis"):
    with st.spinner(f"📥 Fetching {index_name} tickers..."):
        tickers = get_index_tickers(index=index_name)

    with st.spinner("🔎 Filtering tickers with price > $100..."):
        tickers_over_100 = filter_high_price_tickers(tickers, threshold=100)

    with st.spinner("📊 Getting stock changes..."):
        df_changes = get_price_and_volume_change(tickers_over_100, str(start_date), str(end_date))

    st.success("✅ Done! Showing results:")
    fig = plot_interactive_bubble(df_changes, index_name, str(start_date), str(end_date))
    st.plotly_chart(fig, use_container_width=True)
