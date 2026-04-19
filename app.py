import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Analisa Saham", layout="wide")

st.title("📈 Aplikasi Analisa Saham Real-Time")

# Input saham
symbol = st.text_input("Masukkan kode saham (contoh: BBCA.JK, TLKM.JK)", "BBCA.JK")

# Ambil data
data = yf.download(symbol, period="6mo", interval="1d")

if data.empty:
    st.error("Data tidak ditemukan, cek kode saham!")
else:
    # Hitung indikator
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()

    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Harga Saham")
        st.line_chart(data[['Close', 'MA20', 'MA50']])

    with col2:
        st.subheader("📉 RSI")
        st.line_chart(data['RSI'])

    # Info terakhir
    last_price = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]

    st.subheader("📌 Analisa Otomatis")

    st.write(f"Harga terakhir: {last_price:.2f}")
    st.write(f"RSI: {last_rsi:.2f}")

    # Sinyal sederhana
    if last_rsi < 30:
        st.success("🔥 Oversold → Potensi BUY")
    elif last_rsi > 70:
        st.error("⚠️ Overbought → Potensi SELL")
    else:
        st.info("⏳ Netral / Konsolidasi")
