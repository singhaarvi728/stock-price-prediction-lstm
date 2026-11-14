# Stock Price Prediction with LSTM

A reproducible time-series forecasting portfolio project using historical market prices and a Long Short-Term Memory (LSTM) neural network.

> **Provenance:** This is a new, reconstructed portfolio implementation inspired by the project topic in the bachelor's CV. It is not claimed to be the original source code. Predictions are educational experiments, not investment advice.

## Pipeline
1. Download daily historical prices (or load a local CSV).
2. Validate and sort the adjusted closing-price series.
3. Create chronological train/test windows without shuffling.
4. Fit a scaler on training data only.
5. Train an LSTM model.
6. Evaluate on the held-out period using MAE and RMSE.
7. Save predictions and a plot for inspection.

## Stack
Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, yfinance, Matplotlib.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --ticker AAPL --period 5y
```

For an offline run, provide a CSV with a `Date` column and an `Adj Close` (or `Close`) column:
```bash
python src/train.py --csv data/prices.csv
```

## Important caveats
Financial prices are noisy and non-stationary. A low test error does not imply profitable trading. This project does not include transaction costs, portfolio risk, market impact, or a trading strategy.
