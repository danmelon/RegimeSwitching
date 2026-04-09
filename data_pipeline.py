import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def run():

    # ── 1. Download data ──────────────────────────────────────────────
    tickers = {
    "spy":  "^GSPC",
    "qqq":  "QQQ",
    "tlt":  "TLT",
    "gld":  "GLD",
    "vix":  "^VIX"
}
    raw = {name: yf.download(ticker, start="2005-01-01", end="2024-12-31",
                            auto_adjust=True, progress=False)["Close"].squeeze()
        for name, ticker in tickers.items()}


    prices = pd.DataFrame(raw).dropna()

    # ── 2. Log returns ────────────────────────────────────────────────
    returns = np.log(prices / prices.shift(1)).dropna()
    returns.columns = [f"{c}_ret" for c in returns.columns]

    # ── 3. Feature engineering ────────────────────────────────────────
    W = 21  # ~1 trading month

    features = pd.DataFrame(index=returns.index)

    # Rolling volatility (annualised)
    features["vol_21d"]     = returns["spy_ret"].rolling(W).std() * np.sqrt(252)

    # Rolling mean return (momentum proxy)
    features["mom_21d"]     = returns["spy_ret"].rolling(W).mean()

    # Rolling skewness (tail-risk signal)
    features["skew_21d"]    = returns["spy_ret"].rolling(W).skew()

    # VIX level (external fear gauge)
    features["vix_level"]   = prices["vix"].reindex(features.index)

    # Yield-curve proxy: TLT return as bond-equity divergence
    features["bond_ret_21d"] = returns["tlt_ret"].rolling(W).mean()

    # Attach raw SPY return for modelling
    features["spy_ret"]     = returns["spy_ret"]

    # ── 4. Clean ──────────────────────────────────────────────────────
    features = features.dropna()

    # Winsorise at 1st / 99th percentile per column
    for col in features.columns:
        lo, hi = features[col].quantile([0.01, 0.99])
        features[col] = features[col].clip(lo, hi)

    # ── 5. Train / test split ─────────────────────────────────────────
    split_date = "2020-01-01"
    train = features.loc[:split_date]
    test  = features.loc[split_date:]

    # ── 6. Standardise ────────────────────────────────────────────────
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train),
                                index=train.index, columns=train.columns)
    test_scaled  = pd.DataFrame(scaler.transform(test),
                                index=test.index,  columns=test.columns)

    print(f"Train: {train_scaled.shape}  |  Test: {test_scaled.shape}")
    print(train_scaled.describe().round(3))

    return train_scaled, test_scaled, train, test, prices, scaler