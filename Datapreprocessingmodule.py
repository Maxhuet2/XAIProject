import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ta  # pip install ta
import os
import pickle

# ===============================
# 1. Paramètres
# ===============================
TICKERS = ['AAPL', 'GOOG', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
SEQUENCE_LENGTH = 50
SAVE_DIR = "figures"
DATA_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================
# 2. Fonctions de traitement
# ===============================
def download_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    df.dropna(inplace=True)
    return df

def compute_features(df):
    close = df['Close'].astype(float).squeeze()
    high = df['High'].astype(float).squeeze()
    low = df['Low'].astype(float).squeeze()
    volume = df['Volume'].astype(float).squeeze()

    df['Return'] = close.pct_change()
    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    df['MACD'] = ta.trend.MACD(close).macd_diff()
    df['BB_width'] = ta.volatility.BollingerBands(close).bollinger_wband()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=high, low=low, close=close).stoch()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df.dropna(inplace=True)
    return df



def scale_data(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def create_sequences(df, feature_cols, target_col):
    X, y = [], []
    for i in range(len(df) - SEQUENCE_LENGTH):
        X.append(df[feature_cols].iloc[i:i+SEQUENCE_LENGTH].values)
        y.append(df[target_col].iloc[i+SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

def plot_timeseries(df, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['RSI'], label='RSI')
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['BB_width'], label='Bollinger Width')
    plt.plot(df['Stoch_K'], label='Stoch %K')
    plt.title(f"{ticker} - Indicateurs Techniques")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{ticker}_indicators.png")
    plt.close()

# ===============================
# 3. Pipeline principal
# ===============================
all_data = {}
feature_cols = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_width', 'Stoch_K', 'OBV']
target_col = 'Close'

for ticker in TICKERS:
    print(f"\n[INFO] Traitement de {ticker}...")
    df = download_data(ticker)
    df = compute_features(df)
    df, scaler = scale_data(df, feature_cols)
    X, y = create_sequences(df, feature_cols, target_col)
    plot_timeseries(df, ticker)

    all_data[ticker] = {
        "df": df,
        "X": X,
        "y": y,
        "scaler": scaler
    }

    # Sauvegarde en fichier pickle
    with open(f"{DATA_DIR}/{ticker}_sequences.pkl", "wb") as f:
        pickle.dump(all_data[ticker], f)

print("\n[INFO] Traitement terminé. Les séquences sont prêtes pour l'entraînement.")
