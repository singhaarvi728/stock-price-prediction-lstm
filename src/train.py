"""Train and evaluate a univariate LSTM price forecaster."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def load_prices(ticker: str, period: str, csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        frame = pd.read_csv(csv_path)
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise SystemExit("Install yfinance or provide --csv.") from exc
        frame = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if frame.empty:
            raise ValueError("No market data returned. Check ticker/network or provide a CSV.")
        frame = frame.reset_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]
    frame.columns = [str(c).strip() for c in frame.columns]
    price_col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    if price_col not in frame.columns:
        raise ValueError("CSV must contain an 'Adj Close' or 'Close' column.")
    if "Date" not in frame.columns:
        if frame.index.name == "Date":
            frame = frame.reset_index()
        else:
            raise ValueError("Input data must contain a Date column.")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[price_col] = pd.to_numeric(frame[price_col], errors="coerce")
    frame = frame.dropna(subset=["Date", price_col]).sort_values("Date")
    frame = frame.drop_duplicates("Date")
    if len(frame) < 80:
        raise ValueError("At least 80 valid daily observations are required.")
    return frame[["Date", price_col]].rename(columns={price_col: "price"})

def make_windows(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, 0])
        y.append(values[i, 0])
    return np.asarray(X, dtype=np.float32).reshape(-1, lookback, 1), np.asarray(y, dtype=np.float32)

def run(args):
    frame = load_prices(args.ticker, args.period, args.csv)
    split = int(len(frame) * (1 - args.test_size))
    if split <= args.lookback or len(frame) - split < 5:
        raise ValueError("Not enough observations for this lookback/test split.")
    # Fit scaling on training observations only to avoid test leakage.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(frame[["price"]].iloc[:split])
    combined_scaled = scaler.transform(frame[["price"]])
    X, y = make_windows(combined_scaled, args.lookback)
    target_indices = np.arange(args.lookback, len(frame))
    train_mask = target_indices < split
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    if not len(X_test):
        raise ValueError("Test set is empty; reduce lookback or test size.")
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install requirements.txt in a compatible Python environment.") from exc
    model = keras.Sequential([
        keras.layers.Input(shape=(args.lookback, 1)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    callbacks = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=32,
              validation_split=0.1, callbacks=callbacks, verbose=1, shuffle=False)
    pred_scaled = model.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(pred_scaled).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = frame.iloc[target_indices[~train_mask]][["Date"]].copy()
    result["actual"] = y_true
    result["predicted"] = y_pred
    result.to_csv(out / "predictions.csv", index=False)
    model.save(out / "lstm_model.keras")
    plt.figure(figsize=(11, 5))
    plt.plot(result["Date"], result["actual"], label="Actual")
    plt.plot(result["Date"], result["predicted"], label="Predicted")
    plt.title(f"{args.ticker} — held-out predictions")
    plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "forecast.png", dpi=160)
    print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nSaved artefacts to {out.resolve()}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--csv", default=None, help="Optional CSV containing Date and Close/Adj Close.")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    if args.lookback < 2 or not 0.05 <= args.test_size <= 0.5 or args.epochs < 1:
        parser.error("lookback >= 2, test-size between 0.05 and 0.5, and epochs >= 1 are required.")
    return args

if __name__ == "__main__":
    run(parse_args())
