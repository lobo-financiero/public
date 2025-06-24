import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay

# === CONFIGURATION ===
purchase_date = "2025-06-17"
benchmark = "SPY"
investment_per_stock = 100
PORTFOLIO_SIZE_TOP_N = 10 # Define how many "top" stocks to show in bar charts

# Get current date, adjusted for business days
current_date = datetime.today()
if BDay().is_on_offset(current_date):
    today = current_date.strftime("%Y-%m-%d")
else:
    today = (current_date - BDay(1)).strftime("%Y-%m-%d")

# === TICKER GROUPS FOR FOUR PORTFOLIOS ===
# TODO: Replace these placeholder tickers with your actual model outputs.
# These represent the top N stocks predicted using the price on the REPORT DATE.
tickers_model_a_report = [
    "DE", "VRSN", "GDDY", "AMCR", "NCLH", "CPB", "MRNA", "MTCH", "FICO",
    "WBD", "WBA", "VTRS", "MSCI", "TSLA", "CZR", "BA", "WYNN", "ADBE", "ABNB",
    "AAPL", "PYPL", "DPZ", "CRWD", "CNC", "REG", "META", "DVA", "ADSK", "GNRC",
    "SYK", "ORLY", "LEN", "PODD", "AMGN", "ARE", "AXON", "CDNS", "UBER", "PARA",
    "PLTR", "MGM", "PEG", "GEN", "TRMB", "NTAP", "INTC", "EPAM", "TDG", "ENPH", "BF-B"
]
tickers_model_b_report = [
    "CPB", "WBA", "MGM", "LYV", "MRNA", "VRSN", "DE", "FICO", "GDDY", "PLTR",
    "CZR", "WBD", "ABNB", "ORLY", "LVS", "VTRS", "ADSK", "DELL", "NCLH", "MTCH",
    "BA", "STX", "ARE", "AMCR", "ON", "IVZ", "FTNT", "IRM", "TDG", "DPZ", "CAH",
    "INTC", "LOW", "CRWD", "NVDA", "BKNG", "HOLX", "COIN", "AZO", "LEN", "MSCI",
    "DLTR", "CCL", "YUM", "WYNN", "MCHP", "DVA", "CCI", "ANET", "DAY"
]

# These represent the top N stocks predicted using the LIVE PRICE to calculate real-time upside.
# I've slightly modified them for demonstration purposes.
tickers_model_a_live = [
    "ENPH", "DE", "FICO", "GDDY", "CPB", "AMCR", "VRSN", "MRNA", "ADBE",
    "VTRS", "NCLH", "DPZ", "CZR", "MSCI", "AAPL", "WBA", "DVA", "CNC", "MTCH",
    "ARE", "WYNN", "EPAM", "REG", "ORLY", "PYPL", "PODD", "SYK", "ADSK", "ABNB",
    "LEN", "WBD", "CRWD", "CDNS", "TMUS", "UBER", "BF-B", "ELV", "BA", "IT",
    "NTAP", "MKTX", "TRMB", "GEN", "AMGN", "CAG", "PARA", "PAYC", "PEG", "BBY", "TDG"
]
tickers_model_b_live = [
    "CPB", "ENPH", "LULU", "WBA", "FICO", "MRNA", "MGM", "CZR", "GDDY", "ARE",
    "DPZ", "DE", "ORLY", "VTRS", "LYV", "LOW", "ADSK", "TPL", "ADBE", "YUM",
    "CPAY", "CAG", "DVA", "VRSN", "UNH", "DELL", "CCI", "CLX", "AZO", "SBAC",
    "STZ", "ABNB", "AMCR", "MCD", "HUM", "TMUS", "EPAM", "LEN", "TDG", "ELV",
    "NCLH", "CNC", "IT", "FTNT", "MHK", "MOH", "PCG", "UBER", "MSCI", "VRSK"
]


# Combine all unique tickers for a single data fetch
all_tickers_to_fetch = list(set(
    tickers_model_a_report + tickers_model_b_report +
    tickers_model_a_live + tickers_model_b_live + [benchmark]
))


# === STREAMLIT SETUP ===
st.set_page_config(page_title="Sirius Tracker", layout="wide")
st.title("🌌 Sirius Portfolio Tracker")
st.markdown(f"Tracking real returns from **{purchase_date}** to **{today}**")


# === FMP PRICE FETCHER (CACHED) ===
@st.cache_data(ttl=43200) # Cache data for 12 hours
def fetch_fmp_price_history(symbol, from_date, to_date):
    """Fetches historical close prices for a given symbol."""
    api_key = st.secrets["FMP_API_KEY"]
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&to={to_date}&apikey={api_key}"
    try:
        res = requests.get(url)
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
    """Calculates the average return for a given list of symbols."""
    vals = [returns.get(s, 0) for s in symbols]
    return sum(vals) / len(vals) if vals else 0

def create_bar_chart(title, tickers, all_model_tickers):
    """Creates a Plotly bar chart for a given model's returns."""
    st.subheader(title)
    tickers_top_n = tickers[:PORTFOLIO_SIZE_TOP_N]

    top_n_ret = portfolio_return(tickers_top_n)
    total_ret = portfolio_return(all_model_tickers)
    spy = price_data.get(benchmark, pd.Series())
    spy_return = ((spy.iloc[-1] - spy.iloc[0]) / spy.iloc[0]) * 100 if not spy.empty else 0

    bar_labels = tickers_top_n + [f"🔝 Top {PORTFOLIO_SIZE_TOP_N}", "📦 Total Portfolio", f"📈 {benchmark}"]
    bar_returns = [returns.get(t, 0) for t in tickers_top_n] + [top_n_ret, total_ret, spy_return]
    bar_colors = (
        ["#97E956" if r > 0 else "#F44A46" for r in bar_returns[:PORTFOLIO_SIZE_TOP_N]] +
        ["#057DC9", "#288CFF", "orange"]
    )
    fig = go.Figure(data=[go.Bar(
        x=bar_labels, y=bar_returns, marker_color=bar_colors,
        text=[f"{r:.1f}%" for r in bar_returns], textposition="outside"
    )])
    fig.update_layout(
        template="plotly_dark", title=f"Returns Since {purchase_date}",
        yaxis_title="Return (%)", height=550
    )
    st.plotly_chart(fig, use_container_width=True)

def calculate_portfolio_evolution(symbols):
    """Calculates the total value of a portfolio over time."""
    portfolio_values = []
    for symbol in symbols:
        price_series = price_data.get(symbol)
        if price_series is not None and not price_series.empty:
            buy_price = price_series.iloc[0]
            shares = investment_per_stock / buy_price
            portfolio_values.append(shares * price_series)
    if not portfolio_values: return pd.Series(dtype=float)
    return pd.concat(portfolio_values, axis=1).sum(axis=1)


# === 1. SIDE-BY-SIDE MODEL BREAKDOWNS ===
st.header("📊 Model A: Report Price vs. Live Price Predictions")
col1a, col2a = st.columns(2)
with col1a:
    create_bar_chart("Model A (Report Price)", tickers_model_a_report, tickers_model_a_report)
with col2a:
    create_bar_chart("Model A (Live Price)", tickers_model_a_live, tickers_model_a_live)

st.header("📊 Model B: Report Price vs. Live Price Predictions")
col1b, col2b = st.columns(2)
with col1b:
    create_bar_chart("Model B (Report Price)", tickers_model_b_report, tickers_model_b_report)
with col2b:
    create_bar_chart("Model B (Live Price)", tickers_model_b_live, tickers_model_b_live)

# === 2. COMBINED PORTFOLIO PERFORMANCE PLOT ===
st.header("📈 Combined Portfolio Value Evolution")

# Calculate evolution for all portfolios
evo_a_report = calculate_portfolio_evolution(tickers_model_a_report)
evo_a_live = calculate_portfolio_evolution(tickers_model_a_live)
evo_b_report = calculate_portfolio_evolution(tickers_model_b_report)
evo_b_live = calculate_portfolio_evolution(tickers_model_b_live)
# For SPY, scale the initial investment to match an average portfolio size for better visual comparison
avg_portfolio_size = (len(tickers_model_a_report) + len(tickers_model_b_report)) / 2
evo_spy = calculate_portfolio_evolution([benchmark]) * avg_portfolio_size

fig_line = go.Figure()
if not evo_a_report.empty: fig_line.add_trace(go.Scatter(x=evo_a_report.index, y=evo_a_report, mode='lines', name='Model A (Report)', line=dict(color='royalblue', dash='solid')))
if not evo_a_live.empty: fig_line.add_trace(go.Scatter(x=evo_a_live.index, y=evo_a_live, mode='lines', name='Model A (Live)', line=dict(color='royalblue', dash='dash')))
if not evo_b_report.empty: fig_line.add_trace(go.Scatter(x=evo_b_report.index, y=evo_b_report, mode='lines', name='Model B (Report)', line=dict(color='mediumseagreen', dash='solid')))
if not evo_b_live.empty: fig_line.add_trace(go.Scatter(x=evo_b_live.index, y=evo_b_live, mode='lines', name='Model B (Live)', line=dict(color='mediumseagreen', dash='dash')))
if not evo_spy.empty: fig_line.add_trace(go.Scatter(x=evo_spy.index, y=evo_spy, mode='lines', name='SPY (Benchmark)', line=dict(color='orange', width=3)))

fig_line.update_layout(template="plotly_dark", title="Portfolio Growth Over Time", yaxis_title="Total Portfolio Value ($)", legend_title="Portfolio")
st.plotly_chart(fig_line, use_container_width=True)


# === 3. UNIFIED STOCK MEMBERSHIP TABLE ===
st.header("📋 Master Stock List")
all_portfolio_tickers = sorted(list(set(
    tickers_model_a_report + tickers_model_b_report +
    tickers_model_a_live + tickers_model_b_live
)))

table_data = []
for symbol in all_portfolio_tickers:
    if symbol in price_data:
        table_data.append({
            "Symbol": symbol,
            "Current Price": price_data[symbol].iloc[-1],
            "Return %": returns.get(symbol, 0),
            "In Model A (Report)": symbol in tickers_model_a_report,
            "In Model A (Live)": symbol in tickers_model_a_live,
            "In Model B (Report)": symbol in tickers_model_b_report,
            "In Model B (Live)": symbol in tickers_model_b_live,
        })

if table_data:
    membership_df = pd.DataFrame(table_data)
    st.dataframe(
        membership_df,
        use_container_width=True,
        column_config={
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "In Model A (Report)": st.column_config.CheckboxColumn(disabled=True),
            "In Model A (Live)": st.column_config.CheckboxColumn(disabled=True),
            "In Model B (Report)": st.column_config.CheckboxColumn(disabled=True),
            "In Model B (Live)": st.column_config.CheckboxColumn(disabled=True),
        },
        hide_index=True
    )
