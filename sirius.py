#@title Sirius Portfolio Tracker (Dashboard Code with 3 Models)

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay

# === CONFIGURATION ===

# --- 1. Define initial date strings ---
purchase_date_str = "2025-06-18"
benchmark = "SPY"
investment_per_stock = 100
PORTFOLIO_SIZE_TOP_N = 10 # Define how many "top" stocks to show

# --- 2. Create a reusable, robust function for date adjustment ---
def adjust_to_previous_bday(date_obj):
    """
    Checks if a date is a business day. If not, rolls it back to the previous one.
    This is essential for ensuring we query dates with actual market data.
    """
    # The BDay().is_on_offset() checks if the date is a business day (not weekend/holiday)
    if not BDay().is_on_offset(date_obj):
        # If it's not a business day, subtract 1 business day to get the previous one.
        return date_obj - BDay()
    # If it is already a business day, return it unchanged.
    return date_obj

# --- 3. Process and adjust the purchase_date ---
# Convert the string to a pandas Timestamp, which is more powerful than datetime.
purchase_date_dt = pd.to_datetime(purchase_date_str)
# Adjust it to the last valid business day if needed.
adjusted_purchase_date_dt = adjust_to_previous_bday(purchase_date_dt)
# Finally, convert the adjusted date object back to a string for the API call.
purchase_date = adjusted_purchase_date_dt.strftime("%Y-%m-%d")


# --- 4. Process and adjust the current_date (THE CRITICAL FIX) ---
# Step A: Get the current time in UTC, the universal standard.
utc_now = pd.Timestamp.now(tz='UTC')
# Step B: Convert it to the US/Eastern timezone to align with market reality.
# This solves the "off-by-one-day" problem.
eastern_now = utc_now.tz_convert('US/Eastern')
# Step C: Use the same adjustment function to handle weekends or holidays.
adjusted_today_dt = adjust_to_previous_bday(eastern_now)
# Step D: Convert the final, correct date back to a string for the API call.
today = adjusted_today_dt.strftime("%Y-%m-%d")

# === TICKER GROUPS FOR THREE MODELS ===

# --- Model A (Original #1) ---
tickers_model_a_report = [
    "DE", "VRSN", "GDDY", "AMCR", "NCLH", "CPB", "MRNA", "MTCH", "FICO",
    "WBD", "WBA", "VTRS", "MSCI", "TSLA", "CZR", "BA", "WYNN", "ADBE", "ABNB",
    "AAPL", "PYPL", "DPZ", "CRWD", "CNC", "REG", "META", "DVA", "ADSK", "GNRC",
    "SYK", "ORLY", "LEN", "PODD", "AMGN", "ARE", "AXON", "CDNS", "UBER", "PARA",
    "PLTR", "MGM", "PEG", "GEN", "TRMB", "NTAP", "INTC", "EPAM", "TDG", "ENPH", "BF-B"
]
tickers_model_a_live = [
    "ENPH", "DE", "FICO", "GDDY", "CPB", "AMCR", "VRSN", "MRNA", "ADBE",
    "VTRS", "NCLH", "DPZ", "CZR", "MSCI", "AAPL", "WBA", "DVA", "CNC", "MTCH",
    "ARE", "WYNN", "EPAM", "REG", "ORLY", "PYPL", "PODD", "SYK", "ADSK", "ABNB",
    "LEN", "WBD", "CRWD", "CDNS", "TMUS", "UBER", "BF-B", "ELV", "BA", "IT",
    "NTAP", "MKTX", "TRMB", "GEN", "AMGN", "CAG", "PARA", "PAYC", "PEG", "BBY", "TDG"
]

# --- Model B (Original #2) ---
tickers_model_b_report = [
    "CPB", "WBA", "MGM", "LYV", "MRNA", "VRSN", "DE", "FICO", "GDDY", "PLTR",
    "CZR", "WBD", "ABNB", "ORLY", "LVS", "VTRS", "ADSK", "DELL", "NCLH", "MTCH",
    "BA", "STX", "ARE", "AMCR", "ON", "IVZ", "FTNT", "IRM", "TDG", "DPZ", "CAH",
    "INTC", "LOW", "CRWD", "NVDA", "BKNG", "HOLX", "COIN", "AZO", "LEN", "MSCI",
    "DLTR", "CCL", "YUM", "WYNN", "MCHP", "DVA", "CCI", "ANET", "DAY"
]
tickers_model_b_live = [
    "CPB", "ENPH", "LULU", "WBA", "FICO", "MRNA", "MGM", "CZR", "GDDY", "ARE",
    "DPZ", "DE", "ORLY", "VTRS", "LYV", "LOW", "ADSK", "TPL", "ADBE", "YUM",
    "CPAY", "CAG", "DVA", "VRSN", "UNH", "DELL", "CCI", "CLX", "AZO", "SBAC",
    "STZ", "ABNB", "AMCR", "MCD", "HUM", "TMUS", "EPAM", "LEN", "TDG", "ELV",
    "NCLH", "CNC", "IT", "FTNT", "MHK", "MOH", "PCG", "UBER", "MSCI", "VRSK"
]

# --- Model C (The New, Backtested & Corrected Model) ---
# TODO: Replace these placeholder tickers with your actual new model outputs.
tickers_model_c_report = tickers = [
    "CPB", "ENPH", "MRNA", "CZR", "DE", "NCLH", "TSLA", "MCHP", "PEG", "GDDY",
    "GNRC", "MGM", "MOS", "KKR", "WBD", "WBA", "AMCR", "EW", "SYK", "MU",
    "KLAC", "NVDA", "VTRS", "PLTR", "DLTR", "SNPS", "AAPL", "INTC", "COIN", "LEN",
    "PYPL", "PARA", "CF", "DPZ", "ARE", "NUE", "ADBE", "ORLY", "EA", "VRSN",
    "AMAT", "MTCH", "ADSK", "ADI", "NTAP", "WYNN", "MHK", "AVGO", "F", "SWKS"
]
tickers_model_c_live = [
    "ENPH", "CPB", "MRNA", "CZR", "DE", "NCLH", "GDDY", "PEG", "LULU", "MCHP",
    "TSLA", "SYK", "AMCR", "MGM", "GNRC", "KKR", "AAPL", "DPZ", "EW", "VTRS",
    "ADBE", "ARE", "WBA", "LEN", "SNPS", "MHK", "MOS", "ORLY", "PYPL", "PARA",
    "NVDA", "INTC", "STLD", "WBD", "ADSK", "DLTR", "IPG", "NTAP", "EA", "AVGO",
    "HUM", "DVA", "FICO", "LOW", "F", "HPQ", "EPAM", "ELV", "WYNN", "KMX"
]


# Combine all unique tickers for a single data fetch
all_tickers_to_fetch = list(set(
    tickers_model_a_report + tickers_model_b_report + tickers_model_c_report +
    tickers_model_a_live + tickers_model_b_live + tickers_model_c_live + [benchmark]
))

# === STREAMLIT SETUP ===
st.set_page_config(page_title="Sirius Tracker", layout="wide")
st.title("🌌 Sirius Portfolio Tracker")
st.markdown(f"Tracking real returns from **{purchase_date}** to **{today}**")

# === FMP PRICE FETCHER (CACHED) ===
#@st.cache_data(ttl=43200)
def fetch_fmp_price_history(symbol, from_date, to_date):
    api_key = st.secrets.get("FMP_API_KEY", "")
    if not api_key: st.error("FMP_API_KEY secret not found!"); return pd.Series(dtype=float)
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&to={to_date}&apikey={api_key}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        hist = res.json().get("historical", [])
        if not hist: return pd.Series(dtype=float)
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df.sort_index()["close"]
    except Exception:
        return pd.Series(dtype=float)

# Fetch all data at once
progress_bar = st.progress(0, text="📡 Fetching price data...")
price_data = {}
errors = []
for i, symbol in enumerate(all_tickers_to_fetch):
    s = fetch_fmp_price_history(symbol, purchase_date, today)
    if not s.empty: price_data[symbol] = s
    else: errors.append(symbol)
    progress_bar.progress((i + 1) / len(all_tickers_to_fetch), text=f"📡 Fetching {symbol}...")
progress_bar.empty()
if errors: st.error(f"❌ Failed to fetch data for: {', '.join(errors)}")
else: st.success("✅ All price data loaded successfully")

# === CALCULATE INDIVIDUAL STOCK RETURNS ===
returns = {}
for symbol, s in price_data.items():
    if symbol == benchmark or s.empty: continue
    buy_price = s.iloc[0]
    current_price = s.iloc[-1]
    value = (investment_per_stock / buy_price) * current_price
    returns[symbol] = ((value - investment_per_stock) / investment_per_stock) * 100

# === REUSABLE FUNCTIONS ===
def portfolio_return(symbols):
    vals = [returns.get(s, 0) for s in symbols if s in returns]
    return sum(vals) / len(vals) if vals else 0

def create_bar_chart(title, tickers, all_model_tickers):
    st.subheader(title)
    tickers_top_n = tickers[:PORTFOLIO_SIZE_TOP_N]
    top_n_ret = portfolio_return(tickers_top_n)
    total_ret = portfolio_return(all_model_tickers)
    spy_series = price_data.get(benchmark)
    spy_return = ((spy_series.iloc[-1] - spy_series.iloc[0]) / spy_series.iloc[0]) * 100 if spy_series is not None and not spy_series.empty else 0
    bar_labels = tickers_top_n + [f"🔝 Top {PORTFOLIO_SIZE_TOP_N}", "📦 Total Portfolio", f"📈 {benchmark}"]
    bar_returns = [returns.get(t, 0) for t in tickers_top_n] + [top_n_ret, total_ret, spy_return]
    bar_colors = (["#97E956" if r > spy_return else "#F44A46" for r in bar_returns[:PORTFOLIO_SIZE_TOP_N]] + ["#057DC9", "#288CFF", "orange"])
    fig = go.Figure(data=[go.Bar(x=bar_labels, y=bar_returns, marker_color=bar_colors, text=[f"{r:.1f}%" for r in bar_returns], textposition="outside")])
    fig.update_layout(template="plotly_dark", title=f"Returns Since {purchase_date}", yaxis_title="Return (%)", height=550)
    st.plotly_chart(fig, use_container_width=True)

def calculate_portfolio_gain_pct(symbols):
    if not symbols: return pd.Series(dtype=float)
    initial_investment = len(symbols) * investment_per_stock
    portfolio_values = []
    for symbol in symbols:
        price_series = price_data.get(symbol)
        if price_series is not None and not price_series.empty:
            buy_price = price_series.iloc[0]
            if buy_price > 0:
                shares = investment_per_stock / buy_price
                portfolio_values.append(shares * price_series)
    if not portfolio_values: return pd.Series(dtype=float)
    total_value_over_time = pd.concat(portfolio_values, axis=1).sum(axis=1)
    gain_pct_series = ((total_value_over_time - initial_investment) / initial_investment) * 100
    return gain_pct_series

# === 1. SIDE-BY-SIDE MODEL BREAKDOWNS ===
st.header("📊 Model A (Original #1)")
col1a, col2a = st.columns(2)
with col1a: create_bar_chart("Model A (Report Price)", tickers_model_a_report, tickers_model_a_report)
with col2a: create_bar_chart("Model A (Live Price)", tickers_model_a_live, tickers_model_a_live)

st.header("📊 Model B (Original #2)")
col1b, col2b = st.columns(2)
with col1b: create_bar_chart("Model B (Report Price)", tickers_model_b_report, tickers_model_b_report)
with col2b: create_bar_chart("Model B (Live Price)", tickers_model_b_live, tickers_model_b_live)

# NEW: Added charts for Model C
st.header("📊 Model C (New Backtested Model)")
col1c, col2c = st.columns(2)
with col1c: create_bar_chart("Model C (Report Price)", tickers_model_c_report, tickers_model_c_report)
with col2c: create_bar_chart("Model C (Live Price)", tickers_model_c_live, tickers_model_c_live)

# === 2. COMBINED PORTFOLIO PERFORMANCE PLOT ===
st.header("📈 Portfolio Gain (%) Evolution")

# Calculate evolution for all portfolios
gain_a_report = calculate_portfolio_gain_pct(tickers_model_a_report)
gain_a_live = calculate_portfolio_gain_pct(tickers_model_a_live)
gain_b_report = calculate_portfolio_gain_pct(tickers_model_b_report)
gain_b_live = calculate_portfolio_gain_pct(tickers_model_b_live)
gain_c_report = calculate_portfolio_gain_pct(tickers_model_c_report)
gain_c_live = calculate_portfolio_gain_pct(tickers_model_c_live)

gain_a_report_top10 = calculate_portfolio_gain_pct(tickers_model_a_report[:PORTFOLIO_SIZE_TOP_N])
gain_a_live_top10 = calculate_portfolio_gain_pct(tickers_model_a_live[:PORTFOLIO_SIZE_TOP_N])
gain_b_report_top10 = calculate_portfolio_gain_pct(tickers_model_b_report[:PORTFOLIO_SIZE_TOP_N])
gain_b_live_top10 = calculate_portfolio_gain_pct(tickers_model_b_live[:PORTFOLIO_SIZE_TOP_N])
gain_c_report_top10 = calculate_portfolio_gain_pct(tickers_model_c_report[:PORTFOLIO_SIZE_TOP_N])
gain_c_live_top10 = calculate_portfolio_gain_pct(tickers_model_c_live[:PORTFOLIO_SIZE_TOP_N])

spy_prices = price_data.get(benchmark)
gain_spy = pd.Series(dtype=float)
if spy_prices is not None and not spy_prices.empty:
    gain_spy = (spy_prices / spy_prices.iloc[0] - 1) * 100

# Create the plot
fig_line = go.Figure()

# --- Define the new, robust color and style scheme ---
color_map = {
    'A': 'royalblue',
    'B': 'mediumseagreen',
    'C': 'crimson',
    'SPY': 'orange'
}

# Restructured to be explicit and avoid conflicts
style_map = {
    'full_report':   {'dash': 'solid', 'width': 3},
    'full_live':     {'dash': 'dashdot', 'width': 3},
    'top10_report':  {'dash': 'dash', 'width': 2},
    'top10_live':    {'dash': 'longdash', 'width': 2},
    'benchmark':     {'dash': 'dot', 'width': 3}
}


# --- Add Traces using the new, clean scheme ---

# Model C (New Backtested)
if not gain_c_report.empty: fig_line.add_trace(go.Scatter(x=gain_c_report.index, y=gain_c_report, mode='lines', name='Model C (Report)', line=dict(color=color_map['C'], **style_map['full_report'])))
if not gain_c_live.empty: fig_line.add_trace(go.Scatter(x=gain_c_live.index, y=gain_c_live, mode='lines', name='Model C (Live)', line=dict(color=color_map['C'], **style_map['full_live'])))
if not gain_c_report_top10.empty: fig_line.add_trace(go.Scatter(x=gain_c_report_top10.index, y=gain_c_report_top10, mode='lines', name=f'Model C (Top {PORTFOLIO_SIZE_TOP_N})', line=dict(color=color_map['C'], **style_map['top10_report'])))

# Model B (Original)
if not gain_b_report.empty: fig_line.add_trace(go.Scatter(x=gain_b_report.index, y=gain_b_report, mode='lines', name='Model B (Report)', line=dict(color=color_map['B'], **style_map['full_report'])))
if not gain_b_live.empty: fig_line.add_trace(go.Scatter(x=gain_b_live.index, y=gain_b_live, mode='lines', name='Model B (Live)', line=dict(color=color_map['B'], **style_map['full_live'])))
if not gain_b_report_top10.empty: fig_line.add_trace(go.Scatter(x=gain_b_report_top10.index, y=gain_b_report_top10, mode='lines', name=f'Model B (Top {PORTFOLIO_SIZE_TOP_N})', line=dict(color=color_map['B'], **style_map['top10_report'])))

# Model A (Original)
if not gain_a_report.empty: fig_line.add_trace(go.Scatter(x=gain_a_report.index, y=gain_a_report, mode='lines', name='Model A (Report)', line=dict(color=color_map['A'], **style_map['full_report'])))
if not gain_a_live.empty: fig_line.add_trace(go.Scatter(x=gain_a_live.index, y=gain_a_live, mode='lines', name='Model A (Live)', line=dict(color=color_map['A'], **style_map['full_live'])))
if not gain_a_report_top10.empty: fig_line.add_trace(go.Scatter(x=gain_a_report_top10.index, y=gain_a_report_top10, mode='lines', name=f'Model A (Top {PORTFOLIO_SIZE_TOP_N})', line=dict(color=color_map['A'], **style_map['top10_report'])))

# Benchmark
if not gain_spy.empty: fig_line.add_trace(go.Scatter(x=gain_spy.index, y=gain_spy, mode='lines', name='SPY (Benchmark)', line=dict(color=color_map['SPY'], **style_map['benchmark'])))


# --- Update Layout ---
fig_line.update_layout(
    template="plotly_dark",
    title="Portfolio Growth Over Time",
    yaxis_title="Total Gain (%)",
    legend_title="Portfolio",
    height=600
)

st.plotly_chart(fig_line, use_container_width=True)
# === 3. UNIFIED STOCK MEMBERSHIP TABLE ===
st.header("📋 Master Stock List")
all_portfolio_tickers = sorted(list(set(
    tickers_model_a_report + tickers_model_a_live +
    tickers_model_b_report + tickers_model_b_live +
    tickers_model_c_report + tickers_model_c_live
)))

table_data = []
for symbol in all_portfolio_tickers:
    if symbol in price_data and symbol in returns:
        table_data.append({
            "Symbol": symbol, "Current Price": price_data[symbol].iloc[-1], "Return %": returns[symbol],
            "In Model A (Report)": symbol in tickers_model_a_report, "In Model A (Live)": symbol in tickers_model_a_live,
            "In Model B (Report)": symbol in tickers_model_b_report, "In Model B (Live)": symbol in tickers_model_b_live,
            "In Model C (Report)": symbol in tickers_model_c_report, "In Model C (Live)": symbol in tickers_model_c_live,
        })

if table_data:
    membership_df = pd.DataFrame(table_data).sort_values("Return %", ascending=False)
    st.dataframe(
        membership_df, use_container_width=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Return %": st.column_config.ProgressColumn(format="%.2f%%", min_value=-100, max_value=max(100, membership_df["Return %"].max())),
            "In Model A (Report)": st.column_config.CheckboxColumn(disabled=True), "In Model A (Live)": st.column_config.CheckboxColumn(disabled=True),
            "In Model B (Report)": st.column_config.CheckboxColumn(disabled=True), "In Model B (Live)": st.column_config.CheckboxColumn(disabled=True),
            "In Model C (Report)": st.column_config.CheckboxColumn(disabled=True), "In Model C (Live)": st.column_config.CheckboxColumn(disabled=True),
        },
        hide_index=True, height=600
    )
