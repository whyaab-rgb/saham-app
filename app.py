import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Analisa Saham Real-Time", layout="wide")

st.title("📈 Aplikasi Analisa Saham Real-Time")
st.caption("Analisa teknikal otomatis untuk membantu screening saham. Bukan nasihat keuangan.")

# =========================
# INPUT
# =========================
col_input1, col_input2, col_input3 = st.columns([2, 1, 1])

with col_input1:
    symbol = st.text_input("Masukkan kode saham", "BBCA.JK").upper().strip()

with col_input2:
    period = st.selectbox(
        "Periode data",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=2
    )

with col_input3:
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "1mo"],
        index=0
    )

if not symbol:
    st.warning("Silakan masukkan kode saham terlebih dahulu.")
    st.stop()

# =========================
# AMBIL DATA
# =========================
@st.cache_data(ttl=300)
def load_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=False)
    return data

data = load_data(symbol, period, interval)

# Rapikan kolom bila MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    st.error("Data tidak ditemukan. Cek kembali kode saham, misalnya BBCA.JK, TLKM.JK, BMRI.JK.")
    st.stop()

required_cols = ["Open", "High", "Low", "Close", "Volume"]
for col in required_cols:
    if col not in data.columns:
        st.error(f"Kolom {col} tidak tersedia dari server.")
        st.stop()

data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()

# =========================
# INDIKATOR
# =========================
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# RSI 14
delta = data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema12 - ema26
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["Histogram"] = data["MACD"] - data["Signal"]

# Bollinger Bands
data["BB_MID"] = data["Close"].rolling(20).mean()
std20 = data["Close"].rolling(20).std()
data["BB_UPPER"] = data["BB_MID"] + 2 * std20
data["BB_LOWER"] = data["BB_MID"] - 2 * std20

# Support / resistance sederhana
support = data["Low"].tail(20).min()
resistance = data["High"].tail(20).max()

# =========================
# DATA TERAKHIR
# =========================
last_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else last_close
last_rsi = float(data["RSI"].iloc[-1]) if pd.notna(data["RSI"].iloc[-1]) else None
last_macd = float(data["MACD"].iloc[-1]) if pd.notna(data["MACD"].iloc[-1]) else None
last_signal = float(data["Signal"].iloc[-1]) if pd.notna(data["Signal"].iloc[-1]) else None
last_ma20 = float(data["MA20"].iloc[-1]) if pd.notna(data["MA20"].iloc[-1]) else None
last_ma50 = float(data["MA50"].iloc[-1]) if pd.notna(data["MA50"].iloc[-1]) else None
last_volume = float(data["Volume"].iloc[-1]) if pd.notna(data["Volume"].iloc[-1]) else 0.0
avg_volume20 = float(data["Volume"].tail(20).mean()) if len(data) >= 20 else float(data["Volume"].mean())

# =========================
# SCORING REKOMENDASI
# =========================
score = 0
reasons = []

# Trend
if last_ma20 and last_ma50:
    if last_close > last_ma20 > last_ma50:
        score += 2
        reasons.append("Trend naik: harga di atas MA20 dan MA50")
    elif last_close < last_ma20 < last_ma50:
        score -= 2
        reasons.append("Trend turun: harga di bawah MA20 dan MA50")

# RSI
if last_rsi is not None:
    if last_rsi < 30:
        score += 2
        reasons.append("RSI oversold (<30), ada peluang rebound")
    elif 30 <= last_rsi <= 45:
        score += 1
        reasons.append("RSI masih relatif sehat untuk akumulasi")
    elif last_rsi > 70:
        score -= 2
        reasons.append("RSI overbought (>70), rawan koreksi")

# MACD
if last_macd is not None and last_signal is not None:
    if last_macd > last_signal:
        score += 1
        reasons.append("MACD di atas signal, momentum cenderung bullish")
    else:
        score -= 1
        reasons.append("MACD di bawah signal, momentum cenderung melemah")

# Volume
if last_volume > avg_volume20:
    score += 1
    reasons.append("Volume di atas rata-rata 20 hari")
else:
    reasons.append("Volume belum di atas rata-rata 20 hari")

# Posisi terhadap resistance / support
if resistance > support:
    range_pos = (last_close - support) / (resistance - support)
    if range_pos < 0.3:
        score += 1
        reasons.append("Harga dekat area support")
    elif range_pos > 0.85:
        score -= 1
        reasons.append("Harga dekat area resistance")

# Final recommendation
if score >= 4:
    recommendation = "BUY"
    recommendation_text = "Sinyal teknikal cenderung kuat untuk beli / akumulasi."
    recommendation_color = "success"
elif score >= 2:
    recommendation = "ACCUMULATE"
    recommendation_text = "Sinyal cukup positif, cocok dipantau untuk akumulasi bertahap."
    recommendation_color = "info"
elif score >= 0:
    recommendation = "HOLD"
    recommendation_text = "Sinyal campuran, lebih aman menunggu konfirmasi."
    recommendation_color = "warning"
else:
    recommendation = "SELL / AVOID"
    recommendation_text = "Sinyal teknikal cenderung lemah atau rawan koreksi."
    recommendation_color = "error"

# =========================
# METRICS
# =========================
chg_pct = ((last_close - prev_close) / prev_close * 100) if prev_close != 0 else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Harga Terakhir", f"{last_close:,.2f}", f"{chg_pct:.2f}%")
m2.metric("RSI", f"{last_rsi:.2f}" if last_rsi is not None else "-")
m3.metric("MACD", f"{last_macd:.2f}" if last_macd is not None else "-")
m4.metric("Support", f"{support:,.2f}")
m5.metric("Resistance", f"{resistance:,.2f}")

# =========================
# REKOMENDASI
# =========================
st.subheader("📌 Rekomendasi Otomatis")

if recommendation_color == "success":
    st.success(f"Rekomendasi: {recommendation} — {recommendation_text}")
elif recommendation_color == "info":
    st.info(f"Rekomendasi: {recommendation} — {recommendation_text}")
elif recommendation_color == "warning":
    st.warning(f"Rekomendasi: {recommendation} — {recommendation_text}")
else:
    st.error(f"Rekomendasi: {recommendation} — {recommendation_text}")

with st.expander("Lihat alasan rekomendasi"):
    for i, reason in enumerate(reasons, 1):
        st.write(f"{i}. {reason}")
    st.write(f"**Skor total:** {score}")

# =========================
# CHART CANDLESTICK
# =========================
st.subheader("🕯️ Candlestick Chart")

fig_candle = go.Figure()

fig_candle.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick"
))

fig_candle.add_trace(go.Scatter(
    x=data.index,
    y=data["MA20"],
    mode="lines",
    name="MA20"
))

fig_candle.add_trace(go.Scatter(
    x=data.index,
    y=data["MA50"],
    mode="lines",
    name="MA50"
))

fig_candle.add_trace(go.Scatter(
    x=data.index,
    y=data["BB_UPPER"],
    mode="lines",
    name="BB Upper",
    line=dict(dash="dot")
))

fig_candle.add_trace(go.Scatter(
    x=data.index,
    y=data["BB_LOWER"],
    mode="lines",
    name="BB Lower",
    line=dict(dash="dot")
))

fig_candle.update_layout(
    height=600,
    xaxis_title="Tanggal",
    yaxis_title="Harga",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig_candle, use_container_width=True)

# =========================
# CHART VOLUME
# =========================
st.subheader("📦 Volume")

fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(
    x=data.index,
    y=data["Volume"],
    name="Volume"
))
fig_vol.update_layout(
    height=300,
    xaxis_title="Tanggal",
    yaxis_title="Volume"
)
st.plotly_chart(fig_vol, use_container_width=True)

# =========================
# RSI + MACD
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=30, line_dash="dash")
    fig_rsi.update_layout(height=300, yaxis_title="RSI")
    st.plotly_chart(fig_rsi, use_container_width=True)

with col2:
    st.subheader("📊 MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal"], mode="lines", name="Signal"))
    fig_macd.add_trace(go.Bar(x=data.index, y=data["Histogram"], name="Histogram"))
    fig_macd.update_layout(height=300, yaxis_title="MACD")
    st.plotly_chart(fig_macd, use_container_width=True)

# =========================
# DATA TERAKHIR
# =========================
st.subheader("🧾 Data Terakhir")
display_cols = ["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD", "Signal"]
st.dataframe(data[display_cols].tail(10), use_container_width=True)

# =========================
# CATATAN
# =========================
st.caption(
    "Rekomendasi otomatis ini berbasis indikator teknikal sederhana "
    "(MA, RSI, MACD, volume, support/resistance). Gunakan bersama analisa fundamental dan manajemen risiko."
)
