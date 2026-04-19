import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Analisa Saham Real-Time", layout="wide")

# =========================================
# WATCHLIST SAHAM
# =========================================
DEFAULT_STOCKS = [
    "BBCA.JK", "BMRI.JK", "BBRI.JK", "TLKM.JK", "ASII.JK",
    "ICBP.JK", "INDF.JK", "MDKA.JK", "ANTM.JK", "PGAS.JK",
    "ADRO.JK", "ITMG.JK", "UNTR.JK", "EXCL.JK", "CPIN.JK"
]

# =========================================
# HELPER
# =========================================
@st.cache_data(ttl=300)
def load_stock_data(symbol, period="6mo", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return df

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return calculate_indicators(df)

def calculate_indicators(df):
    df = df.copy()

    # Moving Averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # EMA
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Bollinger Bands
    df["BB_MID"] = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * std20
    df["BB_LOWER"] = df["BB_MID"] - 2 * std20

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Volume MA
    df["VOL_MA5"] = df["Volume"].rolling(5).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # Support / Resistance sederhana
    df["SUPPORT20"] = df["Low"].rolling(20).min()
    df["RESIST20"] = df["High"].rolling(20).max()

    # Return harian
    df["RET"] = df["Close"].pct_change() * 100

    return df

def safe_float(series):
    try:
        return float(series.iloc[-1])
    except:
        return None

# =========================================
# LOGIKA STRATEGI
# =========================================
def analyze_scalping(df):
    score = 0
    reasons = []

    close = safe_float(df["Close"])
    ema9 = safe_float(df["EMA9"])
    ma20 = safe_float(df["MA20"])
    rsi = safe_float(df["RSI"])
    macd = safe_float(df["MACD"])
    signal = safe_float(df["MACD_SIGNAL"])
    vol = safe_float(df["Volume"])
    vol_ma5 = safe_float(df["VOL_MA5"])
    resist = safe_float(df["RESIST20"])

    if close and ema9 and ma20:
        if close > ema9 > ma20:
            score += 3
            reasons.append("Harga di atas EMA9 dan MA20, momentum cepat naik")
        elif close < ema9 < ma20:
            score -= 2
            reasons.append("Harga di bawah EMA9 dan MA20")

    if rsi is not None:
        if 55 <= rsi <= 70:
            score += 2
            reasons.append("RSI mendukung momentum scalping")
        elif rsi > 75:
            score -= 1
            reasons.append("RSI terlalu tinggi, rawan profit taking")

    if macd is not None and signal is not None:
        if macd > signal:
            score += 2
            reasons.append("MACD di atas signal")
        else:
            score -= 1
            reasons.append("MACD di bawah signal")

    if vol and vol_ma5:
        if vol > vol_ma5:
            score += 2
            reasons.append("Volume di atas rata-rata 5 hari")
        else:
            score -= 1
            reasons.append("Volume belum kuat")

    if close and resist:
        if close >= resist * 0.98:
            score += 1
            reasons.append("Harga dekat area breakout")

    return score, reasons

def analyze_bsjp(df):
    # Asumsi BSJP = Buy Saat Jenuh Penurunan / buy on weakness
    score = 0
    reasons = []

    close = safe_float(df["Close"])
    bb_lower = safe_float(df["BB_LOWER"])
    ma20 = safe_float(df["MA20"])
    rsi = safe_float(df["RSI"])
    macd_hist = safe_float(df["MACD_HIST"])
    support = safe_float(df["SUPPORT20"])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else close

    if rsi is not None:
        if rsi < 35:
            score += 3
            reasons.append("RSI rendah, saham mulai jenuh turun")
        elif 35 <= rsi <= 45:
            score += 1
            reasons.append("RSI masih di area bawah")

    if close and bb_lower:
        if close <= bb_lower * 1.03:
            score += 2
            reasons.append("Harga dekat lower Bollinger Band")

    if close and support:
        if close <= support * 1.05:
            score += 2
            reasons.append("Harga dekat support 20 hari")

    if macd_hist is not None:
        if macd_hist > 0:
            score += 2
            reasons.append("Histogram MACD mulai positif, ada tanda rebound")
        else:
            reasons.append("Histogram MACD belum positif")

    if close and prev_close:
        if close > prev_close:
            score += 1
            reasons.append("Harga mulai rebound dari penurunan")

    if close and ma20 and close > ma20:
        score -= 1
        reasons.append("Harga sudah jauh dari area buy on weakness")

    return score, reasons

def analyze_swing(df):
    score = 0
    reasons = []

    close = safe_float(df["Close"])
    ma20 = safe_float(df["MA20"])
    ma50 = safe_float(df["MA50"])
    rsi = safe_float(df["RSI"])
    macd = safe_float(df["MACD"])
    signal = safe_float(df["MACD_SIGNAL"])
    vol = safe_float(df["Volume"])
    vol_ma20 = safe_float(df["VOL_MA20"])
    resist = safe_float(df["RESIST20"])
    support = safe_float(df["SUPPORT20"])

    if close and ma20 and ma50:
        if close > ma20 > ma50:
            score += 3
            reasons.append("Trend swing naik: Close > MA20 > MA50")
        elif close < ma20 < ma50:
            score -= 2
            reasons.append("Trend swing turun")

    if rsi is not None:
        if 50 <= rsi <= 65:
            score += 2
            reasons.append("RSI sehat untuk swing")
        elif rsi > 75:
            score -= 1
            reasons.append("RSI terlalu tinggi")

    if macd is not None and signal is not None:
        if macd > signal:
            score += 2
            reasons.append("MACD bullish crossover / di atas signal")
        else:
            score -= 1
            reasons.append("MACD melemah")

    if vol and vol_ma20:
        if vol > vol_ma20:
            score += 1
            reasons.append("Volume mendukung tren")

    if close and support and resist and resist > support:
        pos = (close - support) / (resist - support)
        if 0.4 <= pos <= 0.8:
            score += 1
            reasons.append("Posisi harga masih ideal untuk swing")
        elif pos > 0.9:
            score -= 1
            reasons.append("Harga sudah dekat resistance")

    return score, reasons

def analyze_bandarmology(df):
    # Pendekatan sederhana, bukan broker summary asli
    score = 0
    reasons = []

    close = safe_float(df["Close"])
    ma20 = safe_float(df["MA20"])
    vol = safe_float(df["Volume"])
    vol_ma20 = safe_float(df["VOL_MA20"])
    resist = safe_float(df["RESIST20"])

    recent_up_days = (df["RET"].tail(10) > 0).sum()
    recent_down_days = (df["RET"].tail(10) < 0).sum()

    # Money flow proxy sederhana
    acc_dist_score = 0
    recent = df.tail(10).copy()
    for _, row in recent.iterrows():
        if row["Close"] > row["Open"] and row["Volume"] > row["VOL_MA20"]:
            acc_dist_score += 1
        elif row["Close"] < row["Open"] and row["Volume"] > row["VOL_MA20"]:
            acc_dist_score -= 1

    if acc_dist_score >= 3:
        score += 3
        reasons.append("Terlihat akumulasi: candle naik dengan volume besar lebih dominan")
    elif acc_dist_score <= -3:
        score -= 3
        reasons.append("Terlihat distribusi: candle turun dengan volume besar lebih dominan")

    if close and ma20:
        if close > ma20:
            score += 1
            reasons.append("Harga bertahan di atas MA20")

    if vol and vol_ma20:
        if vol > vol_ma20 * 1.2:
            score += 2
            reasons.append("Lonjakan volume di atas rata-rata 20 hari")

    if close and resist:
        if close >= resist * 0.99:
            score += 1
            reasons.append("Harga mendekati / menembus resistance, potensi markup")

    if recent_up_days > recent_down_days:
        score += 1
        reasons.append("Hari hijau lebih dominan dalam 10 sesi terakhir")

    return score, reasons

def get_recommendation(score):
    if score >= 7:
        return "SANGAT MENARIK"
    elif score >= 5:
        return "MENARIK"
    elif score >= 3:
        return "CUKUP MENARIK"
    elif score >= 1:
        return "PANTAU"
    else:
        return "KURANG MENARIK"

# =========================================
# SCREENING
# =========================================
@st.cache_data(ttl=300)
def screen_stocks(symbols, period="6mo", interval="1d"):
    rows = []

    for symbol in symbols:
        try:
            df = load_stock_data(symbol, period, interval)
            if df.empty or len(df) < 60:
                continue

            close = safe_float(df["Close"])
            rsi = safe_float(df["RSI"])
            vol = safe_float(df["Volume"])

            s_score, _ = analyze_scalping(df)
            b_score, _ = analyze_bsjp(df)
            sw_score, _ = analyze_swing(df)
            bd_score, _ = analyze_bandarmology(df)

            total_score = s_score + b_score + sw_score + bd_score

            rows.append({
                "Symbol": symbol,
                "Close": round(close, 2) if close else None,
                "RSI": round(rsi, 2) if rsi is not None else None,
                "Volume": int(vol) if vol is not None else None,
                "Scalping": s_score,
                "BSJP": b_score,
                "Swing": sw_score,
                "Bandarmologi": bd_score,
                "Total": total_score,
                "Rekomendasi": get_recommendation(total_score)
            })
        except:
            continue

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
    return result

# =========================================
# SIDEBAR
# =========================================
menu = st.sidebar.radio("Pilih Menu", ["Dashboard", "Analisa Saham"])

# =========================================
# DASHBOARD
# =========================================
if menu == "Dashboard":
    st.title("📊 Dashboard Utama")
    st.subheader("Aplikasi Analisa Saham Multi-Strategi")

    st.markdown("""
    Aplikasi ini membagi analisa saham menjadi beberapa strategi:

    **1. Scalping**  
    Fokus pada momentum cepat, breakout, volume, dan RSI jangka pendek.

    **2. BSJP**  
    Asumsi logika: *buy saat jenuh penurunan / buy on weakness* menggunakan RSI rendah,
    lower Bollinger Band, support, dan tanda rebound.

    **3. Swing**  
    Fokus pada tren menengah menggunakan MA20, MA50, MACD, dan volume.

    **4. Bandarmologi Sederhana**  
    Pendekatan akumulasi/distribusi berbasis harga dan volume.
    """)

    st.info("Masuk ke menu **Analisa Saham** untuk melihat ranking saham terbaik dan chart detail.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategi", "4")
    c2.metric("Screening", "Otomatis")
    c3.metric("Data", "Yahoo Finance")
    c4.metric("Mode", "Real-Time")

# =========================================
# ANALISA SAHAM
# =========================================
elif menu == "Analisa Saham":
    st.title("📈 Analisa Saham Multi-Strategi")

    col_a, col_b, col_c = st.columns([2, 1, 1])

    with col_a:
        selected_symbol = st.selectbox("Pilih saham untuk analisa detail", DEFAULT_STOCKS, index=0)

    with col_b:
        period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)

    with col_c:
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    st.markdown("### 🏆 Rekomendasi Saham Tertinggi")

    screening_df = screen_stocks(DEFAULT_STOCKS, period=period, interval=interval)

    if screening_df.empty:
        st.error("Screening tidak berhasil mengambil data saham.")
        st.stop()

    top_n = st.slider("Jumlah rekomendasi yang ditampilkan", 3, min(10, len(screening_df)), 5)
    top_df = screening_df.head(top_n)

    st.dataframe(top_df, use_container_width=True)

    st.markdown("### 📌 Top 3 Saham Terbaik")
    top3 = top_df.head(3)
    cols = st.columns(len(top3))
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            st.metric(row["Symbol"], f"{row['Close']}")
            st.write(f"Total Score: **{row['Total']}**")
            st.write(f"Rekomendasi: **{row['Rekomendasi']}**")

    # DETAIL SAHAM
    df = load_stock_data(selected_symbol, period, interval)
    if df.empty:
        st.error("Data saham detail tidak ditemukan.")
        st.stop()

    last_close = safe_float(df["Close"])
    last_rsi = safe_float(df["RSI"])
    support = safe_float(df["SUPPORT20"])
    resistance = safe_float(df["RESIST20"])
    macd = safe_float(df["MACD"])

    st.markdown(f"## 🔍 Detail Analisa: {selected_symbol}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Harga Terakhir", f"{last_close:,.2f}" if last_close else "-")
    m2.metric("RSI", f"{last_rsi:.2f}" if last_rsi is not None else "-")
    m3.metric("Support", f"{support:,.2f}" if support else "-")
    m4.metric("Resistance", f"{resistance:,.2f}" if resistance else "-")

    # PER STRATEGI
    scalping_score, scalping_reasons = analyze_scalping(df)
    bsjp_score, bsjp_reasons = analyze_bsjp(df)
    swing_score, swing_reasons = analyze_swing(df)
    bandar_score, bandar_reasons = analyze_bandarmology(df)

    st.markdown("## 📋 Analisa per Point Strategi")

    tab1, tab2, tab3, tab4 = st.tabs(["1. Scalping", "2. BSJP", "3. Swing", "4. Bandarmologi"])

    with tab1:
        st.subheader("Scalping")
        st.write("Logika: momentum cepat, volume naik, breakout, RSI dan MACD mendukung.")
        st.metric("Skor Scalping", scalping_score)
        st.write(f"Rekomendasi: **{get_recommendation(scalping_score)}**")
        for i, reason in enumerate(scalping_reasons, 1):
            st.write(f"{i}. {reason}")

    with tab2:
        st.subheader("BSJP")
        st.write("Logika: buy saat jenuh penurunan / buy on weakness, mencari area support dan tanda rebound.")
        st.metric("Skor BSJP", bsjp_score)
        st.write(f"Rekomendasi: **{get_recommendation(bsjp_score)}**")
        for i, reason in enumerate(bsjp_reasons, 1):
            st.write(f"{i}. {reason}")

    with tab3:
        st.subheader("Swing")
        st.write("Logika: mengikuti tren menengah, MA20/MA50, MACD bullish, volume mendukung.")
        st.metric("Skor Swing", swing_score)
        st.write(f"Rekomendasi: **{get_recommendation(swing_score)}**")
        for i, reason in enumerate(swing_reasons, 1):
            st.write(f"{i}. {reason}")

    with tab4:
        st.subheader("Bandarmologi Sederhana")
        st.write("Logika: pendekatan akumulasi/distribusi berbasis harga dan volume. Bukan broker summary asli.")
        st.metric("Skor Bandarmologi", bandar_score)
        st.write(f"Rekomendasi: **{get_recommendation(bandar_score)}**")
        for i, reason in enumerate(bandar_reasons, 1):
            st.write(f"{i}. {reason}")

    total_score = scalping_score + bsjp_score + swing_score + bandar_score
    st.markdown("## ✅ Kesimpulan Akhir")
    st.success(f"Total Score {selected_symbol}: **{total_score}** — **{get_recommendation(total_score)}**")

    # CHART
    st.markdown("## 🕯️ Chart Saham")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], mode="lines", name="BB Upper"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], mode="lines", name="BB Lower"))
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.update_layout(height=300, title="RSI")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with c2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"))
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram"))
        fig_macd.update_layout(height=300, title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)

    st.markdown("## 📄 Data Terakhir")
    show_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA20", "MA50", "RSI", "MACD", "MACD_SIGNAL",
        "BB_UPPER", "BB_LOWER", "SUPPORT20", "RESIST20"
    ]
    st.dataframe(df[show_cols].tail(10), use_container_width=True)

    st.caption(
        "Catatan: hasil screening ini berbasis indikator teknikal otomatis. "
        "Bandarmologi yang digunakan adalah pendekatan sederhana dari harga-volume, "
        "bukan data broker summary asli."
    )
