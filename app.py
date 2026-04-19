import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="High Prob Screener", layout="wide")

# =========================================================
# CONFIG
# =========================================================
DEFAULT_SYMBOLS = [
    "BBCA.JK", "BMRI.JK", "BBRI.JK", "TLKM.JK", "ASII.JK",
    "ICBP.JK", "INDF.JK", "ANTM.JK", "MDKA.JK", "PGAS.JK",
    "ADRO.JK", "ITMG.JK", "UNTR.JK", "AKRA.JK", "ESSA.JK",
    "EXCL.JK", "SMGR.JK", "CPIN.JK", "UNVR.JK", "KLBF.JK",
    "TPIA.JK", "BRPT.JK", "GOTO.JK", "BUKA.JK", "ISAT.JK",
    "MAPI.JK", "ERAA.JK", "ACES.JK", "INKP.JK", "TKIM.JK"
]

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #081018;
    color: white;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 98%;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #081018 0%, #0d1520 100%);
}
[data-testid="stSidebar"] {
    background-color: #0b1220;
}
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #e9f1ff !important;
}
.small-note {
    font-size: 12px;
    color: #9fb4d1 !important;
}
.screen-box {
    border: 1px solid #1b2b43;
    border-radius: 12px;
    padding: 10px;
    background: #09121d;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
}
.screener-title {
    text-align: center;
    font-weight: 700;
    font-size: 14px;
    color: #e9f1ff;
    margin-bottom: 8px;
    letter-spacing: 0.2px;
}
.custom-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    overflow: hidden;
}
.custom-table th {
    background: #183b69;
    color: #ffffff;
    border: 1px solid #274c7a;
    padding: 6px 4px;
    text-align: center;
    white-space: nowrap;
}
.custom-table td {
    border: 1px solid #203550;
    padding: 5px 4px;
    text-align: center;
    white-space: nowrap;
    font-weight: 600;
}
.legend-line {
    margin-top: 6px;
    text-align: center;
    color: #ffd44d !important;
    font-size: 12px;
    font-weight: 700;
}
.footer-line {
    margin-top: 4px;
    text-align: center;
    color: #ffd44d !important;
    font-size: 11px;
}
.metric-card {
    background: #09121d;
    border: 1px solid #1b2b43;
    border-radius: 12px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA SOURCE LAYER
# =========================================================
@st.cache_data(ttl=300)
def get_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df


@st.cache_data(ttl=300)
def get_intraday_5m(symbol: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period="5d",
            interval="5m",
            auto_adjust=False,
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna().copy()
    except Exception:
        return pd.DataFrame()


# =========================================================
# ANALYSIS ENGINE
# =========================================================
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA10"] = x["Close"].rolling(10).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA50"] = x["Close"].rolling(50).mean()
    x["EMA9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()

    delta = x["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    x["RSI"] = 100 - (100 / (1 + rs))

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = ema12 - ema26
    x["MACD_SIGNAL"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["MACD_HIST"] = x["MACD"] - x["MACD_SIGNAL"]

    x["BB_MID"] = x["Close"].rolling(20).mean()
    std20 = x["Close"].rolling(20).std()
    x["BB_UPPER"] = x["BB_MID"] + 2 * std20
    x["BB_LOWER"] = x["BB_MID"] - 2 * std20

    x["VOL_MA5"] = x["Volume"].rolling(5).mean()
    x["VOL_MA20"] = x["Volume"].rolling(20).mean()

    x["SUPPORT20"] = x["Low"].rolling(20).min()
    x["RESIST20"] = x["High"].rolling(20).max()

    high_low = x["High"] - x["Low"]
    high_close = np.abs(x["High"] - x["Close"].shift())
    low_close = np.abs(x["Low"] - x["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    x["ATR14"] = tr.rolling(14).mean()

    x["RET"] = x["Close"].pct_change() * 100

    body = (x["Close"] - x["Open"]).abs()
    upper_wick = x["High"] - x[["Open", "Close"]].max(axis=1)
    lower_wick = x[["Open", "Close"]].min(axis=1) - x["Low"]
    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)

    x["BODY"] = body
    x["UPPER_WICK"] = upper_wick.clip(lower=0)
    x["LOWER_WICK"] = lower_wick.clip(lower=0)
    x["WICK_PCT"] = ((x["UPPER_WICK"] + x["LOWER_WICK"]) / candle_range) * 100

    return x


def latest(series):
    try:
        return float(series.iloc[-1])
    except Exception:
        return np.nan


def fmt_price(v):
    if pd.isna(v):
        return "-"
    return f"{v:,.0f}" if v >= 100 else f"{v:,.2f}"


def fmt_pct(v):
    if pd.isna(v):
        return "-"
    return f"{v:.1f}%"


def transaction_value(close_val, vol_val):
    if pd.isna(close_val) or pd.isna(vol_val):
        return 0.0
    return float(close_val) * float(vol_val)


def human_value(v):
    if v >= 1_000_000_000_000:
        return f"{v/1_000_000_000_000:.1f}T"
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    return f"{v:,.0f}"


def get_rsi_signal(rsi, macd, macd_signal):
    if pd.isna(rsi):
        return "WAIT"
    if rsi >= 60 and macd > macd_signal:
        return "UP"
    if rsi <= 40 and macd < macd_signal:
        return "DEAD"
    if 48 <= rsi <= 55 and macd > macd_signal:
        return "GOLDEN"
    return "UP" if rsi >= 50 else "DEAD"


def get_trend(close_, ma20, ma50):
    if pd.isna(close_) or pd.isna(ma20) or pd.isna(ma50):
        return "NEUTRAL"
    if close_ > ma20 > ma50:
        return "BULL"
    if close_ < ma20 < ma50:
        return "BEAR"
    return "NEUTRAL"


def get_phase(df: pd.DataFrame):
    recent = df.tail(10).copy()
    score = 0
    for _, row in recent.iterrows():
        if pd.notna(row["VOL_MA20"]) and row["VOL_MA20"] > 0:
            if row["Close"] > row["Open"] and row["Volume"] > row["VOL_MA20"]:
                score += 1
            elif row["Close"] < row["Open"] and row["Volume"] > row["VOL_MA20"]:
                score -= 1

    if score >= 4:
        return "BIG AKUM"
    if score >= 2:
        return "AKUM"
    if score <= -4:
        return "BIG DIST"
    if score <= -2:
        return "DIST"
    return "NEUTRAL"


def get_signal_label(close_, ma20, rsi, macd, macd_signal, vol, vol_ma20, support, resistance):
    if any(pd.isna(v) for v in [close_, ma20, rsi, macd, macd_signal]):
        return "WAIT"

    near_support = (not pd.isna(support)) and close_ <= support * 1.05
    near_resistance = (not pd.isna(resistance)) and close_ >= resistance * 0.98
    vol_ok = (not pd.isna(vol_ma20)) and vol > vol_ma20
    bullish = macd > macd_signal

    if near_resistance and bullish and vol_ok and rsi >= 55:
        return "ON TRACK"
    if near_support and rsi < 40 and bullish:
        return "REBOUND"
    if close_ > ma20 and bullish and vol_ok:
        return "AKUM"
    if close_ < ma20 and not bullish and rsi < 45:
        return "DIST"
    if rsi > 68 and bullish and vol_ok:
        return "SUPER"
    return "WAIT"


def get_action_label(signal_label, close_, entry, tp, sl, trend):
    if signal_label in ["ON TRACK", "AKUM", "SUPER", "REBOUND"]:
        if not pd.isna(entry) and close_ <= entry * 1.02:
            return "AT ENTRY"
        if signal_label == "SUPER":
            return "SIAP BELI"
        return "WATCH"
    if trend == "BULL":
        return "HOLD"
    return "WAIT GC"


def compute_strategy_scores(df: pd.DataFrame):
    close_ = latest(df["Close"])
    ma20 = latest(df["MA20"])
    ma50 = latest(df["MA50"])
    ema9 = latest(df["EMA9"])
    rsi = latest(df["RSI"])
    macd = latest(df["MACD"])
    macd_signal = latest(df["MACD_SIGNAL"])
    volume = latest(df["Volume"])
    vol_ma5 = latest(df["VOL_MA5"])
    vol_ma20 = latest(df["VOL_MA20"])
    support = latest(df["SUPPORT20"])
    resistance = latest(df["RESIST20"])
    bb_lower = latest(df["BB_LOWER"])

    # Scalping
    scalping = 0
    if close_ > ema9 > ma20:
        scalping += 3
    if 55 <= rsi <= 72:
        scalping += 2
    if macd > macd_signal:
        scalping += 2
    if volume > vol_ma5:
        scalping += 2
    if close_ >= resistance * 0.98:
        scalping += 1

    # BSJP / Buy Saat Jenuh Penurunan
    bsjp = 0
    if rsi < 35:
        bsjp += 3
    elif 35 <= rsi <= 45:
        bsjp += 1
    if close_ <= bb_lower * 1.03:
        bsjp += 2
    if close_ <= support * 1.05:
        bsjp += 2
    if df["Close"].iloc[-1] > df["Close"].iloc[-2]:
        bsjp += 1
    if df["MACD_HIST"].iloc[-1] > 0:
        bsjp += 2

    # Swing
    swing = 0
    if close_ > ma20 > ma50:
        swing += 3
    if 50 <= rsi <= 65:
        swing += 2
    if macd > macd_signal:
        swing += 2
    if volume > vol_ma20:
        swing += 1
    if close_ < resistance * 0.92 and close_ > support * 1.08:
        swing += 1

    # Bandarmology proxy
    bandarmology = 0
    phase = get_phase(df)
    if phase == "BIG AKUM":
        bandarmology += 4
    elif phase == "AKUM":
        bandarmology += 2
    elif phase == "DIST":
        bandarmology -= 2
    elif phase == "BIG DIST":
        bandarmology -= 4
    if volume > vol_ma20 * 1.2:
        bandarmology += 2
    if close_ > ma20:
        bandarmology += 1

    return {
        "scalping": scalping,
        "bsjp": bsjp,
        "swing": swing,
        "bandarmology": bandarmology
    }


def build_row(symbol: str, daily_df: pd.DataFrame, intraday_5m: pd.DataFrame):
    df = calc_indicators(daily_df)

    if len(df) < 30:
        return None

    close_ = latest(df["Close"])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else close_
    gain = ((close_ - prev_close) / prev_close * 100) if prev_close else 0.0
    wick = latest(df["WICK_PCT"])
    rsi = latest(df["RSI"])
    macd = latest(df["MACD"])
    macd_signal = latest(df["MACD_SIGNAL"])
    vol = latest(df["Volume"])
    vol_ma20 = latest(df["VOL_MA20"])
    close_ma20 = latest(df["MA20"])
    ma50 = latest(df["MA50"])
    support = latest(df["SUPPORT20"])
    resistance = latest(df["RESIST20"])
    atr = latest(df["ATR14"])

    rvol = (vol / vol_ma20 * 100) if vol_ma20 and not pd.isna(vol_ma20) else np.nan

    entry = round((support + close_ma20) / 2) if not pd.isna(support) and not pd.isna(close_ma20) else close_
    now_price = close_
    tp = round(close_ + (atr * 2)) if not pd.isna(atr) else close_ * 1.04
    sl = round(close_ - atr) if not pd.isna(atr) else close_ * 0.97

    profit = ((now_price - entry) / entry * 100) if entry else 0.0
    to_tp = ((tp - now_price) / now_price * 100) if now_price else 0.0

    intraday_rsi = np.nan
    if not intraday_5m.empty and "Close" in intraday_5m.columns:
        intra = calc_indicators(intraday_5m.copy())
        intraday_rsi = latest(intra["RSI"])

    rsi_sig = get_rsi_signal(rsi, macd, macd_signal)
    trend = get_trend(close_, close_ma20, ma50)
    phase = get_phase(df)
    signal_label = get_signal_label(close_, close_ma20, rsi, macd, macd_signal, vol, vol_ma20, support, resistance)
    action_label = get_action_label(signal_label, close_, entry, tp, sl, trend)
    value = transaction_value(close_, vol)

    strat = compute_strategy_scores(df)
    total_score = strat["scalping"] + strat["bsjp"] + strat["swing"] + strat["bandarmology"]

    return {
        "symbol": symbol.replace(".JK", ""),
        "full_symbol": symbol,
        "gain": gain,
        "wick": wick,
        "aksi": action_label,
        "sinyal": signal_label,
        "rvol": rvol,
        "entry": entry,
        "now": now_price,
        "tp": tp,
        "sl": sl,
        "profit": profit,
        "to_tp": to_tp,
        "rsi_sig": rsi_sig,
        "rsi_5m": intraday_rsi,
        "val": value,
        "fase": phase,
        "trend": trend,
        "score_scalping": strat["scalping"],
        "score_bsjp": strat["bsjp"],
        "score_swing": strat["swing"],
        "score_bandar": strat["bandarmology"],
        "score_total": total_score,
        "daily_df": df
    }


@st.cache_data(ttl=300)
def run_screener(symbols, daily_period, daily_interval):
    rows = []
    for symbol in symbols:
        try:
            daily = get_ohlcv(symbol, period=daily_period, interval=daily_interval)
            if daily.empty:
                continue
            intra5 = get_intraday_5m(symbol)
            row = build_row(symbol, daily, intra5)
            if row is not None:
                rows.append(row)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("score_total", ascending=False).reset_index(drop=True)


# =========================================================
# COLOR ENGINE
# =========================================================
def bg_gain(v):
    if pd.isna(v):
        return "#233246"
    if v > 2:
        return "#12b76a"
    if v > 0:
        return "#2d8f61"
    if v > -2:
        return "#d92d20"
    return "#a61b15"


def bg_wick(v):
    if pd.isna(v):
        return "#233246"
    if v < 1:
        return "#0b7a75"
    if v < 2.5:
        return "#1f6feb"
    if v < 4:
        return "#d97706"
    return "#ef4444"


def bg_aksi(v):
    mapping = {
        "AT ENTRY": "#1d4ed8",
        "WATCH": "#b45309",
        "WAIT GC": "#374151",
        "HOLD": "#2563eb",
        "SIAP BELI": "#7c3aed"
    }
    return mapping.get(v, "#334155")


def bg_sinyal(v):
    mapping = {
        "ON TRACK": "#16a34a",
        "REBOUND": "#d97706",
        "AKUM": "#2e8b57",
        "DIST": "#b91c1c",
        "SUPER": "#7e22ce",
        "WAIT": "#111827"
    }
    return mapping.get(v, "#334155")


def bg_rvol(v):
    if pd.isna(v):
        return "#233246"
    if v >= 200:
        return "#8b5cf6"
    if v >= 120:
        return "#f97316"
    if v >= 80:
        return "#1d4ed8"
    return "#374151"


def bg_price(v, ref=None, kind="normal"):
    if pd.isna(v):
        return "#233246"
    if kind == "tp":
        return "#16a34a"
    if kind == "sl":
        return "#b91c1c"
    if kind == "entry":
        return "#1d4ed8"
    if kind == "now":
        return "#2563eb"
    return "#233246"


def bg_profit(v):
    if pd.isna(v):
        return "#233246"
    if v > 2:
        return "#16a34a"
    if v > 0:
        return "#0f766e"
    if v > -2:
        return "#92400e"
    return "#b91c1c"


def bg_to_tp(v):
    if pd.isna(v):
        return "#233246"
    if v <= 1:
        return "#f97316"
    if v <= 3:
        return "#16a34a"
    return "#0f766e"


def bg_rsi_sig(v):
    mapping = {
        "UP": "#16a34a",
        "DEAD": "#dc2626",
        "GOLDEN": "#7c3aed",
        "WAIT": "#111827"
    }
    return mapping.get(v, "#334155")


def bg_rsi(v):
    if pd.isna(v):
        return "#233246"
    if v >= 70:
        return "#f59e0b"
    if v >= 55:
        return "#16a34a"
    if v >= 45:
        return "#2563eb"
    return "#7c3aed"


def bg_fase(v):
    mapping = {
        "BIG AKUM": "#9333ea",
        "AKUM": "#16a34a",
        "NEUTRAL": "#374151",
        "DIST": "#dc2626",
        "BIG DIST": "#b91c1c"
    }
    return mapping.get(v, "#334155")


def bg_trend(v):
    mapping = {
        "BULL": "#16a34a",
        "BEAR": "#dc2626",
        "NEUTRAL": "#6b7280"
    }
    return mapping.get(v, "#334155")


# =========================================================
# TABLE RENDER
# =========================================================
def make_html_table(df: pd.DataFrame, title: str):
    html = f"""
    <div class="screen-box">
        <div class="screener-title">{title}</div>
        <table class="custom-table">
            <thead>
                <tr>
                    <th>EMITEN</th>
                    <th>GAIN</th>
                    <th>WICK</th>
                    <th>AKSI</th>
                    <th>SINYAL</th>
                    <th>RVOL</th>
                    <th>ENTRY</th>
                    <th>NOW</th>
                    <th>TP</th>
                    <th>SL</th>
                    <th>PROFIT</th>
                    <th>%TO TP</th>
                    <th>RSI SIG</th>
                    <th>RSI 5M</th>
                    <th>VAL</th>
                    <th>FASE</th>
                    <th>TREND</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in df.iterrows():
        html += f"""
        <tr>
            <td style="background:#1d4ed8;color:#fff;">{row['symbol']}</td>
            <td style="background:{bg_gain(row['gain'])};color:#fff;">{fmt_pct(row['gain'])}</td>
            <td style="background:{bg_wick(row['wick'])};color:#fff;">{fmt_pct(row['wick'])}</td>
            <td style="background:{bg_aksi(row['aksi'])};color:#fff;">{row['aksi']}</td>
            <td style="background:{bg_sinyal(row['sinyal'])};color:#fff;">{row['sinyal']}</td>
            <td style="background:{bg_rvol(row['rvol'])};color:#fff;">{fmt_pct(row['rvol'])}</td>
            <td style="background:{bg_price(row['entry'], kind='entry')};color:#fff;">{fmt_price(row['entry'])}</td>
            <td style="background:{bg_price(row['now'], kind='now')};color:#fff;">{fmt_price(row['now'])}</td>
            <td style="background:{bg_price(row['tp'], kind='tp')};color:#fff;">{fmt_price(row['tp'])}</td>
            <td style="background:{bg_price(row['sl'], kind='sl')};color:#fff;">{fmt_price(row['sl'])}</td>
            <td style="background:{bg_profit(row['profit'])};color:#fff;">{fmt_pct(row['profit'])}</td>
            <td style="background:{bg_to_tp(row['to_tp'])};color:#fff;">{fmt_pct(row['to_tp'])}</td>
            <td style="background:{bg_rsi_sig(row['rsi_sig'])};color:#fff;">{row['rsi_sig']}</td>
            <td style="background:{bg_rsi(row['rsi_5m'])};color:#fff;">{row['rsi_5m']:.1f if not pd.isna(row['rsi_5m']) else '-'}</td>
            <td style="background:#183b69;color:#fff;">{human_value(row['val'])}</td>
            <td style="background:{bg_fase(row['fase'])};color:#fff;">{row['fase']}</td>
            <td style="background:{bg_trend(row['trend'])};color:#fff;">{row['trend']}</td>
        </tr>
        """

    html += """
            </tbody>
        </table>
        <div class="footer-line">AKSI = tindakan trader | SINYAL = kondisi pasar | SL ≈ 1xATR | TP ≈ 2xATR | versi yfinance</div>
    </div>
    """
    return html


# =========================================================
# SCREENER GROUPING
# =========================================================
def split_screeners(df: pd.DataFrame):
    high_prob = df.sort_values(
        by=["score_total", "score_scalping", "score_swing"],
        ascending=False
    ).head(10)

    rebound = df.sort_values(
        by=["score_bsjp", "score_total"],
        ascending=False
    ).head(10)

    trend = df.sort_values(
        by=["score_swing", "score_bandar", "score_total"],
        ascending=False
    ).head(10)

    return high_prob, rebound, trend


# =========================================================
# DETAIL CHART
# =========================================================
def show_detail_chart(row_df: pd.DataFrame, symbol_name: str):
    st.subheader(f"Chart Detail: {symbol_name}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=row_df.index,
        open=row_df["Open"],
        high=row_df["High"],
        low=row_df["Low"],
        close=row_df["Close"],
        name="Candlestick"
    ))
    fig.add_trace(go.Scatter(x=row_df.index, y=row_df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=row_df.index, y=row_df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=row_df.index, y=row_df["BB_UPPER"], mode="lines", name="BB Upper"))
    fig.add_trace(go.Scatter(x=row_df.index, y=row_df["BB_LOWER"], mode="lines", name="BB Lower"))
    fig.update_layout(
        height=560,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=row_df.index, y=row_df["RSI"], mode="lines", name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.update_layout(height=280, template="plotly_dark", title="RSI")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with c2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=row_df.index, y=row_df["MACD"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=row_df.index, y=row_df["MACD_SIGNAL"], mode="lines", name="Signal"))
        fig_macd.add_trace(go.Bar(x=row_df.index, y=row_df["MACD_HIST"], name="Histogram"))
        fig_macd.update_layout(height=280, template="plotly_dark", title="MACD")
        st.plotly_chart(fig_macd, use_container_width=True)


# =========================================================
# UI
# =========================================================
st.title("HIGH PROB SCREENER V1.0 — YFINANCE VERSION")
st.markdown('<div class="small-note">Tampilan dibuat mirip screener pada gambar, dengan logika berbasis price, volume, RSI, MACD, ATR, support-resistance, dan proxy bandarmologi.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Pengaturan")
    period = st.selectbox("Periode", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    custom_symbols = st.text_area(
        "Daftar saham (pisahkan dengan koma)",
        value=",".join(DEFAULT_SYMBOLS[:15]),
        height=180
    )
    run_btn = st.button("Jalankan Screener", use_container_width=True)

symbols = [x.strip().upper() for x in custom_symbols.split(",") if x.strip()]
if not symbols:
    st.warning("Masukkan minimal 1 kode saham.")
    st.stop()

if run_btn or "loaded_once" not in st.session_state:
    st.session_state["loaded_once"] = True
    with st.spinner("Mengambil data saham dan menghitung screener..."):
        screener_df = run_screener(symbols, period, interval)
        st.session_state["screener_df"] = screener_df
else:
    screener_df = st.session_state.get("screener_df", pd.DataFrame())

if screener_df.empty:
    st.error("Tidak ada data yang berhasil diproses.")
    st.stop()

high_prob_df, rebound_df, trend_df = split_screeners(screener_df)

st.markdown(make_html_table(high_prob_df, "HIGH PROB SCREENER — MOMENTUM & DAY TRADING"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(make_html_table(rebound_df, "HIGH PROB SCREENER — REBOUND / BSJP"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(make_html_table(trend_df, "HIGH PROB SCREENER — SWING / TREND"), unsafe_allow_html=True)

st.markdown("---")
st.subheader("Ranking Saham Tertinggi")

rank_view = screener_df[[
    "symbol", "gain", "rvol", "rsi_5m", "fase", "trend",
    "score_scalping", "score_bsjp", "score_swing", "score_bandar", "score_total"
]].copy()

rank_view.columns = [
    "EMITEN", "GAIN", "RVOL", "RSI 5M", "FASE", "TREND",
    "SCALPING", "BSJP", "SWING", "BANDAR", "TOTAL"
]
st.dataframe(rank_view, use_container_width=True)

selected_symbol = st.selectbox(
    "Pilih saham untuk chart detail",
    screener_df["full_symbol"].tolist()
)

selected_row = screener_df[screener_df["full_symbol"] == selected_symbol].iloc[0]
selected_df = selected_row["daily_df"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("EMITEN", selected_row["symbol"])
c2.metric("TOTAL SCORE", f"{selected_row['score_total']}")
c3.metric("SINYAL", selected_row["sinyal"])
c4.metric("FASE", selected_row["fase"])
c5.metric("TREND", selected_row["trend"])

show_detail_chart(selected_df, selected_row["symbol"])

st.subheader("Detail Analisa Strategi")

t1, t2, t3, t4 = st.tabs(["Scalping", "BSJP", "Swing", "Bandarmologi"])

with t1:
    st.write("Logika scalping: momentum cepat, breakout, volume aktif, MACD bullish, RSI sehat.")
    st.metric("Score Scalping", int(selected_row["score_scalping"]))
    st.write(f"Entry: **{fmt_price(selected_row['entry'])}**")
    st.write(f"TP: **{fmt_price(selected_row['tp'])}**")
    st.write(f"SL: **{fmt_price(selected_row['sl'])}**")
    st.write(f"Aksi: **{selected_row['aksi']}**")

with t2:
    st.write("Logika BSJP: buy saat jenuh penurunan, dekat support/lower band, mulai muncul rebound.")
    st.metric("Score BSJP", int(selected_row["score_bsjp"]))
    st.write(f"RSI 5M: **{selected_row['rsi_5m']:.2f if not pd.isna(selected_row['rsi_5m']) else '-'}**")
    st.write(f"Sinyal: **{selected_row['sinyal']}**")

with t3:
    st.write("Logika swing: trend menengah, MA20/MA50, MACD, volume, dan posisi harga terhadap resistance.")
    st.metric("Score Swing", int(selected_row["score_swing"]))
    st.write(f"Trend: **{selected_row['trend']}**")
    st.write(f"% to TP: **{fmt_pct(selected_row['to_tp'])}**")

with t4:
    st.write("Logika bandarmologi di versi ini adalah proxy price-volume, bukan broker summary asli.")
    st.metric("Score Bandarmologi", int(selected_row["score_bandar"]))
    st.write(f"Fase: **{selected_row['fase']}**")
    st.write(f"Value transaksi: **{human_value(selected_row['val'])}**")

st.caption("Catatan: versi ini dibuat dengan yfinance, sehingga beberapa komponen seperti broker summary, foreign flow, dan orderbook belum tersedia.")
