import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.preprocess import fetch_stock_data, prepare_data, build_model, forecast

st.set_page_config(page_title="Stock Forecast", layout="centered")
st.title("Stock Forecast")

st.markdown("""
### Why Forecast Stock Prices?
Predicting stock trends helps you make better decisions for investment using patterns in historical data. With LSTM, this portal provides model sequential trends for multi-day forecasting.
""")

symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA)", "AAPL")
forecast_days = st.slider("Days to Forecast", min_value=7, max_value=60, value=30)
show_table = st.toggle("Show Forecast Table")

def get_api_key():
    return "e7d3573317054526a18e7ad5fed42d37"  # your Twelve Data API

if st.button("Train & Forecast"):
    try:
        with st.spinner("Fetching stock data..."):
            df = fetch_stock_data(symbol, get_api_key())
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            st.line_chart(df[['close', 'sma_10', 'ema_10']].dropna())

        x, y, scaler = prepare_data(df.dropna())
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

        with st.spinner("Training LSTM model..."):
            model = build_model((x_train.shape[1], 1))
            model.fit(x_train, y_train, epochs=5, batch_size=32)

        with st.spinner("Forecasting future prices..."):
            last_seq = df['close'].dropna().values[-60:]
            last_seq_scaled = scaler.transform(last_seq.reshape(-1, 1)).flatten()
            future = forecast(model, last_seq_scaled, scaler, n_steps=forecast_days)

            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
            future_series = pd.Series(future.flatten(), index=future_dates)

        st.subheader("Forecast Plot")
        fig, ax = plt.subplots()
        df['close'].plot(ax=ax, label='Historical', color='blue')
        future_series.plot(ax=ax, label='Forecast', color='orange', linestyle='--')

        std_dev = np.std(future.flatten())
        ax.fill_between(future_series.index, future_series - std_dev, future_series + std_dev,
                        color='orange', alpha=0.2, label='Confidence Band')
        ax.axvline(x=df.index[-1], color='gray', linestyle=':')
        ax.legend()
        st.pyplot(fig)

        if show_table:
            forecast_df = pd.DataFrame({
                "Date": future_dates.date,
                "Predicted Close": future.flatten(),
                "Confidence Lower": (future.flatten() - std_dev),
                "Confidence Upper": (future.flatten() + std_dev)
            })
            st.subheader("Forecast Table")
            st.dataframe(forecast_df)

        with st.spinner("Evaluating on test set..."):
            test_preds = model.predict(x_test, verbose=0)
            mse = mean_squared_error(y_test, test_preds)
            rmse = np.sqrt(mse)
            st.markdown(f"**RMSE on Test Set**: `{rmse:.4f}`")

        # AI Suggestion (basic logic)
        trend = "🔺 UP" if future[-1] > df['close'].values[-1] else "🔻 DOWN"
        st.subheader("Suggestion based on Analysis")
        st.info(f"Based on current LSTM trend, price movement is expected to be: **{trend}**")

    except Exception as e:
        st.error(f"Error: {e}")

# --- Footer ---
st.markdown(
    """
    <hr style="margin-top: 50px;"/>
    <div style='text-align: center; font-size: 0.9em;'>
        Made by <strong>Licia Saikia</strong>
    </div>
    """,
    unsafe_allow_html=True
)
