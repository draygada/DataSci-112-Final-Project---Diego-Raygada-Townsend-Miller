# S&P 500 Options Screener

A quantitative options screener that combines machine learning, NLP sentiment 
analysis, and options pricing theory to identify mispriced contracts across all 
500 S&P 500 stocks.

## What It Does

The screener runs a full pipeline for every ticker in the S&P 500:

1. **Data Collection** — Pulls 10 years of historical price data via `yfinance`, 
   plus SPY as a market benchmark and VIX as an implied volatility reference.

2. **Feature Engineering** — Builds ~20 predictive features per stock including 
   momentum (5/20/60-day returns), rolling volatility, z-scores, beta, RSI-style 
   signals, EWM momentum, on-balance volume ratio, and relative strength vs. SPY.

3. **Return Forecasting** — Trains a Gradient Boosting model (via scikit-learn) 
   per stock per time horizon to predict forward returns. Predictions are capped 
   at the 95th percentile of historical moves to avoid overconfidence.

4. **Sentiment Analysis** — Uses FinBERT (a finance-specific BERT model) to score 
   recent headlines as positive, negative, or neutral. Flags catalyst events 
   (earnings, FDA decisions, M&A, macro releases) and checks whether sentiment 
   aligns with the proposed trade direction.

5. **Options Pricing** — For each contract in the chain, computes implied 
   volatility via Newton-Raphson on Black-Scholes, probability of finishing 
   in-the-money, expected profit, and return on premium.

6. **Signal Scoring** — Combines move ratio (predicted move vs. implied move), 
   sentiment alignment, catalyst presence, and expected return on premium into a 
   single composite score.

7. **Parallel Execution** — Screens all 500 tickers concurrently using 
   `ThreadPoolExecutor` (8 workers).

## Key Files

- `DataSci_112_FINAL.ipynb` — Main notebook with full pipeline
- `top_options.csv` — Top 25 contracts by signal score
- `total_top_options.csv` — Full results sorted by implied move

## Tech Stack

- **Data**: `yfinance`, `pandas`, `numpy`
- **ML**: `scikit-learn` (GradientBoostingRegressor, StandardScaler, Pipeline)
- **NLP**: `transformers` (ProsusAI/finbert), `torch`
- **Options Math**: Black-Scholes, Newton-Raphson IV solver, `scipy.stats`
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor`
- **Visualization**: `matplotlib`, `seaborn`

## How to Run

1. Open `DataSci_112_FINAL.ipynb` in Google Colab
2. Run all cells top to bottom
3. Results appear printed and saved to CSV

> GPU recommended for FinBERT inference but not required — falls back to CPU automatically.

## Filters Applied

Contracts must pass two filters to appear in results:
- Extrinsic value ≥ 0.5% of stock price (filters out deep ITM options)
- Expected return on premium ≥ 10%
