# Sirius Portfolio Tracker (Dashboard Code with 5 Models, including Vega)

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, time
import plotly.graph_objects as go
from pandas.tseries.offsets import BDay

# === CONFIGURATION (Corrected & Market-Aware) ===

# --- 1. Define initial date strings and constants ---
purchase_date_str = "2025-06-18"
benchmark = "SPY"
investment_per_stock = 100
PORTFOLIO_SIZE_TOP_N = 10
MARKET_CLOSE_TIME = time(16, 15) # Use 4:15 PM to give EOD data time to update.

# --- 2. Create a reusable, robust function for date adjustment ---
def adjust_to_previous_bday(date_obj):
    """Checks if a date is a business day. If not, rolls it back to the previous one."""
    if not BDay().is_on_offset(date_obj):
        return date_obj - BDay()
    return date_obj

# --- 3. Process and adjust the purchase_date ---
purchase_date_dt = pd.to_datetime(purchase_date_str)
adjusted_purchase_date_dt = adjust_to_previous_bday(purchase_date_dt)
purchase_date = adjusted_purchase_date_dt.strftime("%Y-%m-%d")


# --- 4. Process and adjust the current_date (SMART T-1 LOGIC) ---
eastern_now = pd.Timestamp.now(tz='US/Eastern')
if eastern_now.time() > MARKET_CLOSE_TIME:
    final_today_dt = adjust_to_previous_bday(eastern_now)
else:
    final_today_dt = eastern_now - BDay()
today = final_today_dt.strftime("%Y-%m-%d")

st.caption(f"Market data is current as of the end of day: **{today}**")

# === TICKER GROUPS FOR MODELS ===

# --- Model A (Original #1) ---
tickers_model_a_report = ["DE", "VRSN", "GDDY", "AMCR", "NCLH", "CPB", "MRNA", "MTCH", "FICO", "WBD", "WBA", "VTRS", "MSCI", "TSLA", "CZR", "BA", "WYNN", "ADBE", "ABNB", "AAPL", "PYPL", "DPZ", "CRWD", "CNC", "REG", "META", "DVA", "ADSK", "GNRC", "SYK", "ORLY", "LEN", "PODD", "AMGN", "ARE", "AXON", "CDNS", "UBER", "PARA", "PLTR", "MGM", "PEG", "GEN", "TRMB", "NTAP", "INTC", "EPAM", "TDG", "ENPH", "BF-B"]
tickers_model_a_live = ["ENPH", "DE", "FICO", "GDDY", "CPB", "AMCR", "VRSN", "MRNA", "ADBE", "VTRS", "NCLH", "DPZ", "CZR", "MSCI", "AAPL", "WBA", "DVA", "CNC", "MTCH", "ARE", "WYNN", "EPAM", "REG", "ORLY", "PYPL", "PODD", "SYK", "ADSK", "ABNB", "LEN", "WBD", "CRWD", "CDNS", "TMUS", "UBER", "BF-B", "ELV", "BA", "IT", "NTAP", "MKTX", "TRMB", "GEN", "AMGN", "CAG", "PARA", "PAYC", "PEG", "BBY", "TDG"]

# --- Model B (Original #2) ---
tickers_model_b_report = ["CPB", "WBA", "MGM", "LYV", "MRNA", "VRSN", "DE", "FICO", "GDDY", "PLTR", "CZR", "WBD", "ABNB", "ORLY", "LVS", "VTRS", "ADSK", "DELL", "NCLH", "MTCH", "BA", "STX", "ARE", "AMCR", "ON", "IVZ", "FTNT", "IRM", "TDG", "DPZ", "CAH", "INTC", "LOW", "CRWD", "NVDA", "BKNG", "HOLX", "COIN", "AZO", "LEN", "MSCI", "DLTR", "CCL", "YUM", "WYNN", "MCHP", "DVA", "CCI", "ANET", "DAY"]
tickers_model_b_live = ["CPB", "ENPH", "LULU", "WBA", "FICO", "MRNA", "MGM", "CZR", "GDDY", "ARE", "DPZ", "DE", "ORLY", "VTRS", "LYV", "LOW", "ADSK", "TPL", "ADBE", "YUM", "CPAY", "CAG", "DVA", "VRSN", "UNH", "DELL", "CCI", "CLX", "AZO", "SBAC", "STZ", "ABNB", "AMCR", "MCD", "HUM", "TMUS", "EPAM", "LEN", "TDG", "ELV", "NCLH", "CNC", "IT", "FTNT", "MHK", "MOH", "PCG", "UBER", "MSCI", "VRSK"]

# --- Model C1 (New 1-year Model) ---
tickers_model_c1_report = [ "CPB", "NVDA", "DE", "MRNA", "VTRS", "CZR", "MGM", "ENPH", "SMCI", "MCHP", "PARA", "META", "INCY", "WBA", "NCLH", "INTC", "ADSK", "MOS", "KKR", "TSLA", "DECK", "FICO", "GNRC", "PLTR", "ARE", "SWKS", "IVZ", "GDDY", "GEV", "JBL", "BG", "FSLR", "WBD", "CRWD", "CINF", "CRL", "TEL", "IPG", "PLD", "AZO", "CEG", "BBY", "ALB", "F", "EXPE", "MPC", "ON", "DELL", "MU", "LYB" ]
tickers_model_c1_live = [ "CPB", "NVDA", "ENPH", "DE", "MRNA", "LULU", "VTRS", "CZR", "FICO", "ARE", "MGM", "PARA", "ADSK", "IPG", "TPL", "BBY", "DECK", "EPAM", "CAG", "UNH", "NCLH", "ERIE", "WBA", "DPZ", "GDDY", "KKR", "PAYC", "PCG", "AZO", "TMUS", "PODD", "EIX", "IT", "INTC", "KMX", "CNC", "DVA", "CCI", "INCY", "STZ", "HUM", "WSM", "JBL", "HPQ", "F", "LYB", "AVGO", "ALB", "PLD", "UBER" ]

# --- Model C3 (Renamed 3-year Backtested Model) ---
tickers_model_c3_report = ["CPB", "ENPH", "MRNA", "CZR", "DE", "NCLH", "TSLA", "MCHP", "PEG", "GDDY", "GNRC", "MGM", "MOS", "KKR", "WBD", "WBA", "AMCR", "EW", "SYK", "MU", "KLAC", "NVDA", "VTRS", "PLTR", "DLTR", "SNPS", "AAPL", "INTC", "COIN", "LEN", "PYPL", "PARA", "CF", "DPZ", "ARE", "NUE", "ADBE", "ORLY", "EA", "VRSN", "AMAT", "MTCH", "ADSK", "ADI", "NTAP", "WYNN", "MHK", "AVGO", "F", "SWKS"]
tickers_model_c3_live = ["ENPH", "CPB", "MRNA", "CZR", "DE", "NCLH", "GDDY", "PEG", "LULU", "MCHP", "TSLA", "SYK", "AMCR", "MGM", "GNRC", "KKR", "AAPL", "DPZ", "EW", "VTRS", "ADBE", "ARE", "WBA", "LEN", "SNPS", "MHK", "MOS", "ORLY", "PYPL", "PARA", "NVDA", "INTC", "STLD", "WBD", "ADSK", "DLTR", "IPG", "NTAP", "EA", "AVGO", "HUM", "DVA", "FICO", "LOW", "F", "HPQ", "EPAM", "ELV", "WYNN", "KMX"]

# --- Model D (Vega - Your Transformer Model) ---
tickers_vega_report = [ "PYPL", "DLTR", "PNR", "COIN", "SW", "PARA", "AMCR", "GDDY", "ENPH", "OTIS", "AES", "MTCH", "NCLH", "NRG", "GEN", "CNC", "ORLY", "PRU", "MOH", "MCK", "COR", "HPQ", "MOS", "ORCL", "GLW", "FTI", "DPZ", "CTVA", "BA", "DELL", "CI", "HUM", "IVZ", "IRM", "APA", "CAH", "FDS", "IQV", "EQT", "WAT", "CBRE", "LOW", "JNPR", "ELV", "MSCI", "VRSN", "LRCX", "EXPE", "URI", "CHTR" ]
tickers_vega_live = [ "DLTR", "PNR", "PYPL", "GDDY", "COIN", "LULU", "DPZ", "MOH", "TPL", "MCK", "FICO", "COR", "ELV", "CI", "NRG", "AZO", "HUM", "UNH", "TDG", "CPAY", "ERIE", "MSCI", "FDS", "MTD", "NVR", "GWW", "MSI", "WAT", "IT", "LOW", "CHTR", "BKNG", "REGN", "LMT", "POOL", "ROP", "COST", "LII", "PGR", "STZ", "BRK-B", "WTW", "EG", "MCO", "SHW", "ITW", "PH", "URI", "OTIS", "AON" ]


# Combine all unique tickers for a single data fetch
all_tickers_to_fetch = list(set(
    tickers_model_a_report + tickers_model_b_report + tickers_model_c1_report + tickers_model_c3_report + tickers_vega_report +
    tickers_model_a_live + tickers_model_b_live + tickers_model_c1_live + tickers_model_c3_live + tickers_vega_live + [benchmark]
))

# === STREAMLIT SETUP ===
st.set_page_config(page_title="Sirius Tracker", layout="wide")
st.title("🌌 Sirius & Vega Portfolio Tracker")
st.markdown(f"Tracking real returns from **{purchase_date}** to **{today}**")

# === FMP PRICE FETCHER (CACHED) ===
@st.cache_data(ttl=86400)
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
    marker_line_colors, marker_line_widths = [], []
    for r in bar_returns[:PORTFOLIO_SIZE_TOP_N]:
        marker_line_colors.append("lightcoral" if r < 0 else "rgba(0,0,0,0)")
        marker_line_widths.append(2 if r < 0 else 0)
    marker_line_colors.extend(["rgba(0,0,0,0)"] * 3); marker_line_widths.extend([0] * 3)
    fig = go.Figure(data=[go.Bar(x=bar_labels, y=bar_returns, text=[f"{r:.1f}%" for r in bar_returns], textposition="outside", marker=dict(color=bar_colors, line=dict(color=marker_line_colors, width=marker_line_widths)))])
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

st.header("📊 Model C1 (1-year)")
col1c1, col2c1 = st.columns(2)
with col1c1: create_bar_chart("Model C1 (Report Price)", tickers_model_c1_report, tickers_model_c1_report)
with col2c1: create_bar_chart("Model C1 (Live Price)", tickers_model_c1_live, tickers_model_c1_live)

st.header("📊 Model C3 (3-years)")
col1c3, col2c3 = st.columns(2)
with col1c3: create_bar_chart("Model C3 (Report Price)", tickers_model_c3_report, tickers_model_c3_report)
with col2c3: create_bar_chart("Model C3 (Live Price)", tickers_model_c3_live, tickers_model_c3_live)

st.header("📊 Model D (Vega)")
col1d, col2d = st.columns(2)
with col1d: create_bar_chart("Vega (Report Price)", tickers_vega_report, tickers_vega_report)
with col2d: create_bar_chart("Vega (Live Price)", tickers_vega_live, tickers_vega_live)

# === 2. COMBINED PORTFOLIO PERFORMANCE PLOT (REFACTORED WITH FILTERS) ===
st.header("📈 Portfolio Gain (%) Evolution")

# --- STEP 1: Define all portfolios in a structured way ---
portfolio_definitions = {
    "A": {
        "Report Top 50": tickers_model_a_report,
        "Live Top 50": tickers_model_a_live,
        f"Report Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_a_report[:PORTFOLIO_SIZE_TOP_N],
        f"Live Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_a_live[:PORTFOLIO_SIZE_TOP_N],
    },
    "B": {
        "Report Top 50": tickers_model_b_report,
        "Live Top 50": tickers_model_b_live,
        f"Report Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_b_report[:PORTFOLIO_SIZE_TOP_N],
        f"Live Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_b_live[:PORTFOLIO_SIZE_TOP_N],
    },
    "C1": {
        "Report Top 50": tickers_model_c1_report,
        "Live Top 50": tickers_model_c1_live,
        f"Report Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_c1_report[:PORTFOLIO_SIZE_TOP_N],
        f"Live Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_c1_live[:PORTFOLIO_SIZE_TOP_N],
    },
    "C3": {
        "Report Top 50": tickers_model_c3_report,
        "Live Top 50": tickers_model_c3_live,
        f"Report Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_c3_report[:PORTFOLIO_SIZE_TOP_N],
        f"Live Top {PORTFOLIO_SIZE_TOP_N}": tickers_model_c3_live[:PORTFOLIO_SIZE_TOP_N],
    },
    "Vega": {
        "Report Top 50": tickers_vega_report,
        "Live Top 50": tickers_vega_live,
        f"Report Top {PORTFOLIO_SIZE_TOP_N}": tickers_vega_report[:PORTFOLIO_SIZE_TOP_N],
        f"Live Top {PORTFOLIO_SIZE_TOP_N}": tickers_vega_live[:PORTFOLIO_SIZE_TOP_N],
    }
}

# --- STEP 2: Calculate evolution for all defined portfolios (computationally cheap) ---
portfolio_evolutions = {}
for model, types in portfolio_definitions.items():
    portfolio_evolutions[model] = {}
    for type_name, tickers in types.items():
        portfolio_evolutions[model][type_name] = calculate_portfolio_gain_pct(tickers)

spy_prices = price_data.get(benchmark)
gain_spy = pd.Series(dtype=float)
if spy_prices is not None and not spy_prices.empty:
    gain_spy = (spy_prices / spy_prices.iloc[0] - 1) * 100

# --- STEP 3: Add the interactive filters ---
filter_cols = st.columns([1, 1, 2])
with filter_cols[0]:
    selected_models = st.multiselect(
        "Select Model(s)",
        options=list(portfolio_definitions.keys()),
        default=list(portfolio_definitions.keys()) # Default to all models
    )
with filter_cols[1]:
    all_types = list(next(iter(portfolio_definitions.values())).keys())
    selected_types = st.multiselect(
        "Select Portfolio Type(s)",
        options=all_types,
        default=["Live Top 50", "Report Top 50"] # Default to the main portfolios
    )
with filter_cols[2]:
    st.write("") # for vertical alignment
    st.write("")
    show_spy = st.checkbox("Show SPY Benchmark", value=True)


# --- STEP 4: Build the plot dynamically based on selections ---
fig_line = go.Figure()

color_map = {'A': 'royalblue', 'B': 'mediumseagreen', 'C1': 'gold', 'C3': 'crimson', 'Vega': 'darkviolet', 'SPY': 'white'}
style_map = {
    "Report Top 50": {'dash': 'dash', 'width': 3},
    "Live Top 50": {'dash': 'solid', 'width': 3},
    f"Report Top {PORTFOLIO_SIZE_TOP_N}": {'dash': 'dashdot', 'width': 2},
    f"Live Top {PORTFOLIO_SIZE_TOP_N}": {'dash': 'dot', 'width': 2},
}

# Loop through user selections and add traces
for model_name in selected_models:
    for type_name in selected_types:
        if model_name in portfolio_evolutions and type_name in portfolio_evolutions[model_name]:
            series = portfolio_evolutions[model_name][type_name]
            if not series.empty:
                fig_line.add_trace(go.Scatter(
                    x=series.index,
                    y=series,
                    mode='lines',
                    name=f'{model_name} ({type_name})',
                    line=dict(color=color_map.get(model_name), **style_map.get(type_name))
                ))

# Add Benchmark if selected
if show_spy and not gain_spy.empty:
    fig_line.add_trace(go.Scatter(
        x=gain_spy.index,
        y=gain_spy,
        mode='lines',
        name='SPY (Benchmark)',
        line=dict(color=color_map['SPY'], dash='solid', width=3)
    ))

# Update Layout
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
    tickers_model_c1_report + tickers_model_c1_live +
    tickers_model_c3_report + tickers_model_c3_live +
    tickers_vega_report + tickers_vega_live
)))

table_data = []
for symbol in all_portfolio_tickers:
    if symbol in price_data and symbol in returns:
        table_data.append({
            "Symbol": symbol, "Current Price": price_data[symbol].iloc[-1], "Return %": returns[symbol],
            "In Model A (Report)": symbol in tickers_model_a_report, "In Model A (Live)": symbol in tickers_model_a_live,
            "In Model B (Report)": symbol in tickers_model_b_report, "In Model B (Live)": symbol in tickers_model_b_live,
            "In Model C1 (Report)": symbol in tickers_model_c1_report, "In Model C1 (Live)": symbol in tickers_model_c1_live,
            "In Model C3 (Report)": symbol in tickers_model_c3_report, "In Model C3 (Live)": symbol in tickers_model_c3_live,
            "In Vega (Report)": symbol in tickers_vega_report, "In Vega (Live)": symbol in tickers_vega_live,
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
            "In Model C1 (Report)": st.column_config.CheckboxColumn(disabled=True), "In Model C1 (Live)": st.column_config.CheckboxColumn(disabled=True),
            "In Model C3 (Report)": st.column_config.CheckboxColumn(disabled=True), "In Model C3 (Live)": st.column_config.CheckboxColumn(disabled=True),
            "In Vega (Report)": st.column_config.CheckboxColumn(disabled=True), "In Vega (Live)": st.column_config.CheckboxColumn(disabled=True),
        },
        hide_index=True, height=600
    )
