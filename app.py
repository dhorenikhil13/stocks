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
        print(f"❌ Failed to fetch {index} tickers: {e}")
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
                'Price Change %': float(round(price_pct_change, 2)),
                'Volume Change %': float(round(volume_pct_change, 2)),
                'Price': float(round(close_price, 2))
            })

        except Exception as e:
            print(f"Error with {ticker}: {e}")
            continue

    return pd.DataFrame(data)

# === Plotting interactive bubble chart ===
def plot_interactive_bubble(df, index_name, start_date, end_date):
    # Categorize into 6 groups
    def categorize(row):
        p, v = row['Price Change %'], row['Volume Change %']
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

    # Color mapping
    color_map = {
        'Sell - High Vol': '#8B0000',
        'Sell - Low Vol': '#FFA07A',
        'Hold - High Vol': '#00008B',
        'Hold - Low Vol': '#ADD8E6',
        'Star - High Vol': '#006400',
        'Star - Low Vol': '#90EE90'
    }

    df['Color'] = df['Category'].map(color_map)

    # Show text labels only for certain categories
    label_mask = df['Category'].isin(['Sell - High Vol', 'Hold - High Vol', 'Star - High Vol', 'Star - Low Vol'])
    df['Label'] = df['Ticker'].where(label_mask, '')

    fig = px.scatter(
        df,
        x='Price Change %',
        y='Volume Change %',
        color='Category',
        color_discrete_map=color_map,
        hover_data={'Ticker': True, 'Price': True, 'Price Change %': True, 'Volume Change %': True},
        text='Label',
        size_max=30,
        size=[8] * len(df),
        title=f'{index_name} Stocks: Price vs Volume Change<br>({start_date} → {end_date})'
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(title_x=0.5)

    return fig

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📊 Stock Movement Visualizer (Price vs Volume)")

index_name = st.selectbox("Select Index:", ["QQQ", "SPY"])
start_date = st.date_input("Start Date", datetime(2025, 1, 2))
end_date = st.date_input("End Date", datetime(2025, 7, 22))
threshold = st.slider("Minimum Current Price ($)", min_value=0, max_value=500, value=100, step=10)

if st.button("🔍 Analyze"):
    with st.spinner(f"Fetching {index_name} tickers and processing data..."):
        tickers = get_index_tickers(index=index_name)
        if not tickers:
            st.error("Failed to fetch tickers. Please check your internet or ticker source.")
        else:
            tickers_over_threshold = filter_high_price_tickers(tickers, threshold)
            df_changes = get_price_and_volume_change(tickers_over_threshold, str(start_date), str(end_date))

            if df_changes.empty:
                st.warning("No data available after filtering. Try a different date range or threshold.")
            else:
                fig = plot_interactive_bubble(df_changes, index_name, str(start_date), str(end_date))
                st.plotly_chart(fig, use_container_width=True)
