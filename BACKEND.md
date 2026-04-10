# Signal — Backend Documentation

## General Overview

**Trading Signal API** (v2.0.0) is a FastAPI-based backend that ingests financial market data, runs sentiment analysis on financial news, trains a multimodal deep learning model, and generates actionable trading signals (BUY / SELL / HOLD) with associated risk metrics.

**Stack:**
- **API:** FastAPI + Uvicorn
- **ML:** PyTorch (custom Transformer + MLP fusion architecture)
- **Database:** PostgreSQL + TimescaleDB (time-series extension)
- **Sentiment:** HuggingFace FinBERT via Inference Router API
- **Market Data:** Yahoo Finance (yfinance)
- **News:** NewsAPI
- **Scheduler:** APScheduler (post-market automation)
- **Rate Limiting:** slowapi
- **Cache:** In-memory TTL cache with optional Redis backend

**Core output per signal:**
- Action (BUY / SELL / HOLD) + confidence score
- Entry price, stop-loss, take-profit, net profit
- Estimated bars to entry + calendar entry time
- Softmax probabilities for all three classes

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
**Routes:** `POST /api/v1/ingest/`, `POST /api/v1/ingest/ticker`

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
8. Normalize all price features per-feature (mean/std) with **per-ticker Z-score normalization**, save parameters to `scaler_params.json`
9. Chronological **ticker-level** split (no shuffling to prevent look-ahead bias): **70% train / 15% val / 15% test**

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

The best checkpoint (lowest validation loss) is saved to `checkpoints/best_model.pt`. Each successful training run is versioned in the `model_versions` table. Training is tracked via the **Job Service** — the `POST /ml/train` route creates a job record and returns a `job_id` immediately; the background task updates it to `completed` or `failed` on finish.

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

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | `(mean_return / std_return) × √252` — annualized risk-adjusted return |
| Win Rate | % of trades with positive realized return |
| Total Trades | Count of non-HOLD predictions on the test set |
| Avg Return | Mean per-trade return |

---

## Stage 4 — Signal Generation & Inference

**Service:** `app/services/signal_service.py` + `app/services/ml_service.py`  
**Routes:** `POST /api/v1/signals/generate`, `POST /api/v1/signals/generate/batch`

Loads the best checkpoint and generates a trading signal for a ticker using its latest 60 bars and current sentiment snapshot. Predictions are **cached** for 5 minutes (via `CacheService`) to avoid redundant GPU inference.

**Inference steps:**

1. Load `best_model.pt` and `scaler_params.json` from `checkpoints/`
2. Check cache for `predict:{ticker}:{interval}` — return early on hit
3. Query the latest 60 bars of OHLCV + indicators from the database
4. Query the latest sentiment snapshot for the ticker
5. Normalize price features using stored mean/std
6. Run `model.forward(x_price, x_sentiment)` → logits + regression
7. Apply softmax to logits → probabilities for [HOLD, BUY, SELL]
8. Select action = argmax; confidence = max probability
9. Convert `bars_to_entry` to a calendar datetime (next trading day at 14:30 EST for 1d interval)
10. Store result in cache with 300s TTL; store signal in the `signals` table

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
  "entry_time": "2025-10-15T14:30:00-04:00"
}
```

### Signal Explainability

**Route:** `GET /api/v1/signals/{signal_id}/explain`

Uses **gradient-based saliency** (PyTorch `backward()`) to rank which input features drove the model's prediction. Computes the gradient of the predicted class logit with respect to each input feature, averages the absolute gradient across the 60-bar sequence, and returns the top 15 most influential features ranked by importance.

---

## Additional Services

### Cache Service

**File:** `app/services/cache_service.py`

Singleton cache accessed via `get_cache()`. Supports two backends:

| Backend | Description |
|---------|-------------|
| `_MemoryCache` | In-process dict storing `(value, expires_at)` tuples. Default. |
| `_RedisCache` | `redis.asyncio` client with JSON serialization. Enabled when `REDIS_URL` env var is set. |

Used by: ML prediction caching (5-minute TTL).

---

### Job Service

**File:** `app/services/job_service.py`

Tracks long-running background tasks in the `jobs` database table.

| Status | Description |
|--------|-------------|
| `pending` | Job created, not yet started |
| `running` | Background task is active |
| `completed` | Task finished successfully |
| `failed` | Task raised an exception |

Fields: `id`, `job_type`, `status`, `progress` (JSONB), `error`, `started_at`, `finished_at`, `created_at`.

Methods: `create()`, `update_progress()`, `complete()`, `fail()`, `get()`, `get_latest()`.

Used by: ML training (`POST /ml/train`), Walk-Forward validation.

---

### Portfolio Service

**File:** `app/services/portfolio_service.py`

Paper trading engine that tracks simulated positions.

**Features:**
- **Averaging down** — opening a position in a ticker already held computes a new weighted average cost instead of creating a duplicate record
- **Unrealised P&L** — computed on read by joining with the latest close price from `ohlcv_data`
- **Realised P&L** — written to `positions` on close; individual fills recorded in `position_trades`

**Tables:** `positions`, `position_trades`

---

### Scheduler

**File:** `app/services/scheduler.py`

APScheduler jobs that run automatically:

| Job | Schedule | Description |
|-----|----------|-------------|
| `refresh_post_market` | 22:00 UTC, Mon–Fri | Ingests 5-day bars for all tracked tickers after market close |
| `regenerate_signals_post_close` | 22:30 UTC, Mon–Fri | Generates fresh signals for all tracked tickers |
| `check_price_alert_rules` | Every 5 minutes | Compares latest close prices against user-defined alert thresholds; fires alerts and deactivates triggered rules |

---

### Walk-Forward Validation

**Route:** `POST /api/v1/ml/walkforward`

Expanding-window cross-validation across multiple time folds. Each fold trains on a growing prefix and validates on the next unseen segment — this tests the model on market regimes it was never trained on. Returns per-fold and summary metrics (accuracy, F1, Sharpe, win rate, max drawdown).

---

### Signal Outcomes

**Routes:** `GET /api/v1/outcomes`, `GET /api/v1/outcomes/summary`, `POST /api/v1/outcomes/check`

Checks historical signals against actual price movement to assess prediction accuracy. Each checked signal produces an outcome record:

| Outcome | Condition |
|---------|-----------|
| `WIN` | Trade hit take-profit |
| `LOSS` | Trade hit stop-loss |
| `EXPIRED` | Neither hit within the lookahead window |

Fields: `id`, `signal_id`, `ticker`, `action`, `entry_price`, `stop_loss`, `take_profit`, `outcome`, `actual_return`, `bars_held`, `exit_price`, `exit_time`.

---

### Confluence Engine

**Routes:** `GET /api/v1/confluence/{ticker}`, `POST /api/v1/confluence/batch`

Combines signals across multiple timeframes (daily + hourly) into a single confluence score and strength label:

| Strength | Condition |
|----------|-----------|
| `strong` | Both timeframes agree on direction |
| `weak` | One timeframe is HOLD |
| `conflicting` | Timeframes disagree |
| `neutral` | Both HOLD |

Score is normalised to 0–100. Batch endpoint accepts a list of tickers and returns confluence for all.

---

### Backtesting

**Routes:** `GET /api/v1/backtest/{ticker}`, `POST /api/v1/backtest/portfolio`

Replays stored signals against historical prices to produce a simulated equity curve and trade log. Metrics: total return, Sharpe ratio, max drawdown, win rate, per-trade P&L.

---

### User Management & Auth

**Routes:** `GET /auth/me`, `PUT /users/me`, `POST /users/me/password`

JWT-based authentication with **silent refresh token rotation**. On 401, the client deduplicates concurrent refresh calls (single shared promise) to avoid token storms. Profile update allows changing `full_name` and `email`; password change verifies the current password before hashing the new one.

---

### Alerts

**Routes:** `GET /api/v1/alerts`, `GET /api/v1/alerts/count`, `POST /api/v1/alerts/{id}/read`, `POST /api/v1/alerts/read-all`

User-scoped signal alerts (linked to signals table via `signal_id`). Support unread-only filtering, per-alert and bulk mark-as-read.

---

### Price Alert Rules

**Routes:** `GET /api/v1/price-alerts`, `POST /api/v1/price-alerts`, `DELETE /api/v1/price-alerts/{rule_id}`

User-defined threshold triggers. Each rule has a `condition` (`above` / `below`) and `target_price`. Checked every 5 minutes by the scheduler; rule is deactivated once triggered.

---

### Watchlist

**Routes:** `GET /users/me/watchlist`, `POST /users/me/watchlist/{ticker}`, `DELETE /users/me/watchlist/{ticker}`, `PUT /users/me/watchlist`

Per-user list of followed tickers stored as a JSONB array on the `users` table. Supports add, remove, and full replace.

---

### WebSocket Live Prices

**Route:** `GET /api/v1/ws/prices?tickers=AAPL,MSFT`

Streams real-time price updates for subscribed tickers. Message format:

```json
{ "type": "prices", "data": { "AAPL": 213.45 }, "changes": { "AAPL": 0.0023 } }
```

---

## Database Schema (Key Tables)

| Table | Description |
|-------|-------------|
| `ohlcv_data` | Raw OHLCV bars (TimescaleDB hypertable) |
| `indicators` | Computed technical indicators (hypertable) |
| `signals` | Generated trading signals (user-scoped via nullable `user_id`) |
| `sentiment_articles` | Individual news articles with FinBERT scores |
| `sentiment_snapshots` | Aggregated per-ticker sentiment vectors |
| `signal_outcomes` | WIN / LOSS / EXPIRED outcome records |
| `alerts` | User signal notifications (user-scoped via nullable `user_id`) |
| `price_alert_rules` | User-defined price threshold triggers |
| `jobs` | Background job tracking (status, progress, error) |
| `positions` | Paper trading positions with P&L tracking |
| `position_trades` | Individual fill records for position open/close |
| `model_versions` | Versioned model checkpoints with metrics |
| `users` | User accounts with watchlist (JSONB) |

---

## API Quick Reference

### Auth & Users
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/auth/login` | Login (form data) → JWT pair |
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| GET  | `/api/v1/auth/me` | Current user profile |
| PUT  | `/api/v1/users/me` | Update name / email |
| POST | `/api/v1/users/me/password` | Change password |
| GET  | `/api/v1/users/me/watchlist` | Get watchlist |
| POST | `/api/v1/users/me/watchlist/{ticker}` | Add to watchlist |
| DELETE | `/api/v1/users/me/watchlist/{ticker}` | Remove from watchlist |
| PUT  | `/api/v1/users/me/watchlist` | Replace full watchlist |

### Ingest
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/ingest/` | Ingest OHLCV + indicators (≤50 tickers) |
| POST | `/api/v1/ingest/ticker` | Ingest single ticker |
| POST | `/api/v1/ingest/background` | Async ingestion (≤200 tickers) |
| GET  | `/api/v1/ingest/tickers` | List available tickers in DB |
| GET  | `/api/v1/ingest/sp500` | Get S&P 500 ticker list |

### Sentiment
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/sentiment/fetch` | Fetch news & run FinBERT (≤20 tickers) |
| POST | `/api/v1/sentiment/enrich` | Historical sentiment enrichment |
| GET  | `/api/v1/sentiment/summary` | All tickers' sentiment summaries |
| GET  | `/api/v1/sentiment/summary/{ticker}` | Single ticker sentiment summary |
| GET  | `/api/v1/sentiment/snapshot/{ticker}` | Latest sentiment snapshot |
| GET  | `/api/v1/sentiment/snapshot/{ticker}/history` | Snapshot history |
| GET  | `/api/v1/sentiment/articles/{ticker}` | Individual news articles |

### ML
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/ml/train` | Async training → returns `job_id` |
| POST | `/api/v1/ml/train/sync` | Synchronous training (≤20 tickers) |
| GET  | `/api/v1/ml/train/result` | Last training run result |
| POST | `/api/v1/ml/predict` | Single ticker inference (cache-backed) |
| GET  | `/api/v1/ml/report` | Evaluation metrics |
| GET  | `/api/v1/ml/status` | Model checkpoint info |
| POST | `/api/v1/ml/walkforward` | Walk-forward validation |
| GET  | `/api/v1/ml/walkforward/result` | Last walk-forward result |
| GET  | `/api/v1/ml/versions` | List model versions |
| POST | `/api/v1/ml/versions/rollback` | Rollback to a specific version |

### Signals
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/v1/signals/generate` | Generate + store signal for one ticker |
| POST | `/api/v1/signals/generate/batch` | Async batch generation (≤50 tickers) |
| GET  | `/api/v1/signals/latest/{ticker}` | Latest signal for ticker |
| GET  | `/api/v1/signals/history/{ticker}` | Signal history |
| GET  | `/api/v1/signals/filter/action/{action}` | Filter by BUY / SELL / HOLD |
| GET  | `/api/v1/signals/filter/high-confidence` | Signals above confidence threshold |
| GET  | `/api/v1/signals/{signal_id}/explain` | Gradient saliency — top 15 features |
| GET  | `/api/v1/signals/tickers` | List tickers with signals |
| POST | `/api/v1/signals/tickers/override` | Set ticker override list |
| DELETE | `/api/v1/signals/tickers/override` | Clear ticker override |

### Outcomes
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/outcomes` | All outcome records (filterable by ticker) |
| GET  | `/api/v1/outcomes/summary` | Aggregate win/loss/expired stats |
| POST | `/api/v1/outcomes/check` | Trigger outcome check now |

### Alerts
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/alerts` | All alerts (optional unread-only filter) |
| GET  | `/api/v1/alerts/count` | Unread alert count |
| POST | `/api/v1/alerts/{id}/read` | Mark single alert as read |
| POST | `/api/v1/alerts/read-all` | Mark all alerts as read |

### Confluence
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/confluence/{ticker}` | Multi-timeframe confluence score |
| POST | `/api/v1/confluence/batch` | Batch confluence for multiple tickers |

### Backtest
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/backtest/{ticker}` | Single-ticker backtest with equity curve |
| POST | `/api/v1/backtest/portfolio` | Multi-ticker portfolio backtest |

### Portfolio
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/portfolio/positions` | All positions (open + closed) |
| GET  | `/api/v1/portfolio/summary` | Aggregate P&L summary |
| POST | `/api/v1/portfolio/positions` | Open a new position |
| POST | `/api/v1/portfolio/positions/{id}/close` | Close position at exit price |

### Price Alert Rules
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/price-alerts` | All user price alert rules |
| POST | `/api/v1/price-alerts` | Create new price alert rule |
| DELETE | `/api/v1/price-alerts/{rule_id}` | Delete a rule |

### Jobs
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/jobs/{job_id}` | Get job status and progress |
| GET  | `/api/v1/jobs/latest/{job_type}` | Get latest job of a given type |

### Market
| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/v1/market/ohlcv/{ticker}` | OHLCV bars |
| GET  | `/api/v1/market/indicators/{ticker}` | Indicator series |
| GET  | `/api/v1/market/indicators/{ticker}/latest` | Latest indicator snapshot |

### WebSocket
| Route | Description |
|-------|-------------|
| `ws /api/v1/ws/prices?tickers=…` | Live price stream (JSON messages) |
