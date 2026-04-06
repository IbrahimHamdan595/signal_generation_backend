# Signal — Backend Documentation

## General Overview

**Trading Signal API** (v2.0.0) is a FastAPI-based backend that ingests financial market data, runs sentiment analysis on financial news, trains a multimodal deep learning model, and generates actionable trading signals (BUY / SELL / HOLD) with associated risk metrics.

**Stack:**
- **API:** FastAPI
- **ML:** PyTorch (custom Transformer + MLP architecture)
- **Database:** PostgreSQL + TimescaleDB (time-series extension)
- **Sentiment:** HuggingFace FinBERT via Inference Router API
- **Market Data:** Yahoo Finance (yfinance)
- **News:** NewsAPI

**Core output per signal:**
- Action (BUY / SELL / HOLD) + confidence score
- Entry price, stop-loss, take-profit, net profit
- Estimated bars to entry + calendar entry time

---

## Project Stages

```
Yahoo Finance (OHLCV)     NewsAPI (Articles)
        │                       │
        ▼                       ▼
┌───────────────┐     ┌──────────────────────┐
│  Stage 1      │     │  Stage 2             │
│  OHLCV Ingest │     │  Sentiment Analysis  │
└───────┬───────┘     └──────────┬───────────┘
        │                        │
        └──────────┬─────────────┘
                   ▼
          ┌────────────────┐
          │  Stage 3A      │
          │  Dataset Build │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  Stage 3B      │
          │  Model Train   │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  Stage 3C      │
          │  Evaluation    │
          └───────┬────────┘
                  ▼
          ┌────────────────┐
          │  Stage 4       │
          │  Signal Gen    │
          └────────────────┘
```

---

## Stage 1 — OHLCV Ingestion

**Service:** `app/services/ohlcv_service.py`  
**Route:** `POST /api/v1/ingest/`

Fetches raw price data from Yahoo Finance for one or more tickers, computes 50+ technical indicators, and stores everything in PostgreSQL (TimescaleDB hypertables).

**Supported intervals:** `1d`, `1h`, `5m`, `15m`, `30m`  
**Supported periods:** `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`

**Raw OHLCV columns stored (5):**

| Column   | Description        |
|----------|--------------------|
| `open`   | Opening price      |
| `high`   | Daily high         |
| `low`    | Daily low          |
| `close`  | Closing price      |
| `volume` | Trade volume       |

**Technical indicators computed and stored (50):**

| Group       | Indicators |
|-------------|------------|
| Trend       | SMA(20), SMA(50), EMA(12), EMA(26), ADX, +DI, −DI, Pivot (S1, S2, R1, R2) |
| Momentum    | RSI(14), MACD line, MACD signal, MACD histogram, ROC(5), ROC(10), Stochastic K, Stochastic D |
| Volatility  | ATR(14), BB upper, BB middle, BB lower, BB bandwidth, BB position |
| Volume      | OBV, MFI(14), Volume ROC, Volume Above Average |
| Temporal    | day_of_week, day_of_month, month, is_trading_day |
| Derived     | Distance from SMA, volatility regime, price above SMA50, price above SMA200 |
| External    | VIX level, VIX change, earnings flag, social sentiment, put/call ratio (stub) |

---

## Stage 2 — Sentiment Analysis

**Service:** `app/services/sentiment_service.py` + `app/services/finbert_service.py`  
**Route:** `POST /api/v1/sentiment/fetch`

Fetches recent news articles for each ticker and classifies them using **FinBERT** (ProsusAI/finbert via HuggingFace Inference Router). Stores individual article sentiment and computes an aggregated snapshot per ticker.

**Flow:**

1. Fetch up to `NEWS_FETCH_LIMIT` (default 10) articles per ticker from NewsAPI
2. Concatenate article title + description
3. Send each article to FinBERT → receives `label` (positive / negative / neutral) + raw scores
4. Aggregate across articles into a single snapshot vector

**Sentiment snapshot (4 values per ticker):**

| Field          | Description                         |
|----------------|-------------------------------------|
| `avg_positive` | Mean positive score across articles |
| `avg_negative` | Mean negative score across articles |
| `avg_neutral`  | Mean neutral score across articles  |
| `avg_compound` | `avg_positive − avg_negative`       |

---

## Stage 3A — Dataset Construction

**Module:** `app/ml/data/dataset.py`

Builds the training dataset by joining OHLCV, indicators, and sentiment data from the database, then constructing fixed-length sliding windows with labels and regression targets.

**Steps:**

1. Fetch rows from `ohlcv_data` and `indicators` tables per ticker
2. Fetch latest sentiment snapshot per ticker (single 4-value vector reused for all windows)
3. Require at least `seq_len + 2 = 62` rows per ticker; skip tickers below this threshold
4. Select the **18 model features** from the 55 available OHLCV + indicator columns
5. Create sliding windows of length 60 bars
6. Compute classification label for each window (based on return of the *next* close):
   - **BUY (1):** return > +0.5%
   - **SELL (2):** return < −0.5%
   - **HOLD (0):** otherwise
7. Compute 5 regression targets per window using a `lookahead_window = 5` bars and ATR-based risk levels:
   - `entry_price` = current close
   - `stop_loss` = entry ∓ 1.5 × ATR (direction-dependent)
   - `take_profit` = entry ± 3.0 × ATR
   - `net_profit` = |TP − entry| − |SL − entry|
   - `bars_to_entry` = optimal bars until best entry in lookahead window (0–5, clamped)
8. Normalize all price features per-feature (mean/std), save parameters to `scaler_params.json`
9. Chronological split (no shuffling to prevent look-ahead bias): **70% train / 15% val / 15% test**

**Final dataset shapes:**

| Tensor         | Shape       | Description                            |
|----------------|-------------|----------------------------------------|
| `X_price`      | (N, 60, 18) | Price + indicator features per window  |
| `X_sentiment`  | (N, 4)      | Sentiment scores per window            |
| `y_class`      | (N,)        | Classification label (0=HOLD, 1=BUY, 2=SELL) |
| `y_regression` | (N, 5)      | Regression targets per window          |

---

## Stage 3B — Model Training

**Modules:** `app/ml/models/`, `app/ml/training/trainer.py`

### Models

---

### 1. TransformerEncoder

**File:** `app/ml/models/transformer.py`

Encodes the sequence of 60 price/indicator bars into a fixed-size context vector.

| | Shape | Description |
|-|-------|-------------|
| **Input** | `(batch, 60, 18)` | 60-bar window, 18 price/indicator features |
| **Output** | `(batch, 64)` | Mean-pooled sequence embedding |

**Architecture:**

```
Input Projection:  Linear(18 → 64)
Positional Encoding: Sinusoidal, added to projected sequence
Transformer Layers: 2 × TransformerEncoderLayer
  ├─ Self-Attention: 4 heads, d_model=64
  ├─ Feed-Forward:  dim_feedforward=256
  └─ Dropout: 0.1
Mean Pooling: (batch, 60, 64) → (batch, 64)
```

---

### 2. Sentiment Projection

**File:** `app/ml/models/fusion_model.py` (inline module)

Maps the 4-value sentiment snapshot to a 16-dimensional embedding.

| | Shape | Description |
|-|-------|-------------|
| **Input** | `(batch, 4)` | Sentiment scores (pos, neg, neu, compound) |
| **Output** | `(batch, 16)` | Projected sentiment embedding |

**Architecture:**

```
Linear(4 → 16) + ReLU
```

---

### 3. MLPHead

**File:** `app/ml/models/mlp_head.py`

Shared backbone + two output heads (classification and regression) operating on the fused embedding.

| | Shape | Description |
|-|-------|-------------|
| **Input** | `(batch, 80)` | Concatenation of price (64) + sentiment (16) |
| **Output — logits** | `(batch, 3)` | Raw scores for [HOLD, BUY, SELL] |
| **Output — regression** | `(batch, 5)` | [entry_price, stop_loss, take_profit, net_profit, bars_to_entry] |

**Architecture:**

```
Shared MLP:
  Linear(80 → 128) + BatchNorm1d + ReLU + Dropout(0.1)
  Linear(128 → 64) + BatchNorm1d + ReLU + Dropout(0.1)

Classification Head:  Linear(64 → 3)
Regression Head:      Linear(64 → 5)
```

---

### 4. TradingFusionModel (Full Model)

**File:** `app/ml/models/fusion_model.py`

End-to-end model combining the three components above.

| | Shape | Description |
|-|-------|-------------|
| **Input — price** | `(batch, 60, 18)` | Price/indicator sequence |
| **Input — sentiment** | `(batch, 4)` | Aggregated news sentiment |
| **Output — logits** | `(batch, 3)` | Class scores → softmax → probabilities |
| **Output — regression** | `(batch, 5)` | Risk/entry targets |

**Full forward pass:**

```
x_price     → TransformerEncoder  → (batch, 64)   ─┐
x_sentiment → SentimentProjection → (batch, 16)   ─┤
                                   Concat → (batch, 80)
                                   MLPHead
                                   ├─ logits:     (batch, 3)
                                   └─ regression: (batch, 5)
```

**Model configuration** (`checkpoints/model_config.json`):

```json
{
  "n_features": 18,
  "seq_len": 60,
  "d_model": 64,
  "n_heads": 4,
  "n_layers": 2,
  "d_ff": 256,
  "sent_input": 4,
  "sent_dim": 16,
  "mlp_hidden": 128,
  "dropout": 0.1
}
```

**Training setup:**

| Parameter          | Value                         |
|--------------------|-------------------------------|
| Optimizer          | Adam (lr=1e-3, wd=1e-4)       |
| Classification loss| CrossEntropyLoss (weight 1.0) |
| Regression loss    | MSELoss (weight 0.1)          |
| Joint loss         | `1.0 × CE + 0.1 × MSE`        |
| LR Scheduler       | ReduceLROnPlateau (patience=5)|
| Early stopping     | patience=10 epochs            |
| Gradient clipping  | max_norm=1.0                  |
| Max epochs         | 50                            |

The best checkpoint (lowest validation loss) is saved to `checkpoints/best_model.pt`.

---

## Stage 3C — Evaluation

**Module:** `app/ml/evaluation/evaluator.py`  
**Route:** `GET /api/v1/ml/report`

Runs the trained model on the held-out test set (15% of data) and computes classification, regression, and simulated trading metrics. Results are saved to `checkpoints/eval_report.json`.

### Classification Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | % of correctly classified windows |
| F1 (weighted) | F1 weighted by class support |
| F1 (macro) | Unweighted mean F1 across classes |
| Precision / Recall | Per class: HOLD, BUY, SELL |
| Confusion Matrix | 3×3 matrix of true vs. predicted labels |

### Regression Metrics (per target)

| Target | Metrics |
|--------|---------|
| `entry_price` | RMSE, MAE |
| `stop_loss` | RMSE, MAE |
| `take_profit` | RMSE, MAE |
| `net_profit` | RMSE, MAE |

> `bars_to_entry` regression is computed by the model but not separately reported in the eval output.

### Simulated Trading Metrics

These metrics simulate actual trade execution using the model's non-HOLD predictions on the test set.

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | `(mean_return / std_return) × √252` — annualized risk-adjusted return |
| Win Rate | % of trades with positive realized return |
| Total Trades | Count of non-HOLD predictions on the test set |
| Avg Return | Mean per-trade return |

### Running Evaluation

Evaluation runs automatically at the end of every training job. To re-run on demand:

```
GET /api/v1/ml/report
```

Returns the full `eval_report.json` contents including all metrics above.

---

## Stage 4 — Signal Generation & Inference

**Service:** `app/services/signal_service.py` + `app/services/ml_service.py`  
**Routes:** `POST /api/v1/signals/generate`, `POST /api/v1/signals/generate/batch`

Loads the best checkpoint and generates a trading signal for a ticker using its latest 60 bars and current sentiment snapshot.

**Inference steps:**

1. Load `best_model.pt` and `scaler_params.json` from `checkpoints/`
2. Query the latest 60 bars of OHLCV + indicators from the database
3. Query the latest sentiment snapshot for the ticker
4. Normalize price features using stored mean/std
5. Run `model.forward(x_price, x_sentiment)` → logits + regression
6. Apply softmax to logits → probabilities for [HOLD, BUY, SELL]
7. Select action = argmax; confidence = max probability
8. Convert `bars_to_entry` to a calendar datetime (next trading day at 14:30 EST for 1d interval)
9. Store signal in the `signals` table

**Signal output:**

```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "confidence": 0.78,
  "probabilities": { "buy": 0.78, "hold": 0.19, "sell": 0.03 },
  "entry_price": 214.50,
  "stop_loss": 210.20,
  "take_profit": 223.10,
  "net_profit": 4.30,
  "bars_to_entry": 1,
  "entry_time": "2025-10-15T14:30:00-04:00",
  "entry_time_label": "Tomorrow at 2:30 PM EST"
}
```

---

## API Quick Reference

### Ingest
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/ingest/` | Ingest OHLCV + indicators (≤50 tickers) |
| POST | `/api/v1/ingest/background` | Async ingestion (≤200 tickers) |
| GET  | `/api/v1/ingest/tickers` | List available tickers in DB |
| GET  | `/api/v1/ingest/sp500` | Get SP500 ticker list |

### Sentiment
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/sentiment/fetch` | Fetch news & run FinBERT (≤20 tickers) |
| GET  | `/api/v1/sentiment/snapshot/{ticker}` | Latest sentiment snapshot |
| GET  | `/api/v1/sentiment/summary` | All tickers' sentiment summaries |

### ML
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/ml/train` | Async training (2–100 tickers) |
| POST | `/api/v1/ml/train/sync` | Sync training (≤20 tickers) |
| POST | `/api/v1/ml/predict` | Single ticker inference |
| GET  | `/api/v1/ml/report` | Evaluation metrics |
| GET  | `/api/v1/ml/status` | Model checkpoint info |

### Signals
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/signals/generate` | Generate + store signal for one ticker |
| POST | `/api/v1/signals/generate/batch` | Async batch generation (≤50 tickers) |
| GET  | `/api/v1/signals/latest/{ticker}` | Latest signal for ticker |
| GET  | `/api/v1/signals/history/{ticker}` | Signal history |
| GET  | `/api/v1/signals/filter/action/{action}` | Filter by BUY / SELL / HOLD |
| GET  | `/api/v1/signals/filter/high-confidence` | Signals above confidence threshold |
