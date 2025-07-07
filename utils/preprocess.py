import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests


def fetch_stock_data(symbol: str, api_key: str) -> pd.DataFrame:
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&apikey={api_key}&outputsize=5000"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise ValueError("API Error or invalid symbol")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()

    # Standardize all column names to lowercase
    df.columns = df.columns.str.lower()

    if "close" not in df.columns:
        raise ValueError("'close' column is missing in API response")

    df = df.astype(float)
    return df


def prepare_data(df: pd.DataFrame, window_size=60):
    close_prices = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    x, y = [], []
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    x = np.array(x)
    y = np.array(y)
    return x, y, scaler


def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def forecast(model, last_sequence_scaled, scaler, n_steps=30):
    window_size = len(last_sequence_scaled)
    input_seq = np.array(last_sequence_scaled).reshape(1, window_size, 1)
    predictions = []

    for _ in range(n_steps):
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_pred)

        input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

    predicted = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted
