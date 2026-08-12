# Nifty 500 Momentum / Reversal and Trend Analyzer

Streamlit app for an exploratory Indian equity market study across Nifty 50 or Nifty 500.

It finds stocks that made strong 3-day up or down moves, checks what happened over the next 3 and 5 trading days, and adds configurable lookback trend scoring such as 15-day trend strength.

The upgraded version also includes a volume intelligence layer that studies price-volume behaviour, volume spikes, buying/selling pressure proxies, accumulation/distribution patterns, liquidity, and conviction scoring.

## What It Produces

- Automatic conclusion for the presentation
- Top 10 ranking tables for continuation and reversal behavior
- Bullish signal table: strong 3-day up moves and their next 3D / 5D returns
- Bearish signal table: strong 3-day down moves and their next 3D / 5D returns
- Stock-wise summary table with average, median, volatility, best/worst return, and win rate
- Configurable Nifty 50 / Nifty 500 universe selection
- Configurable historical period, defaulting to 3 years
- Configurable trend lookback window, defaulting to 15 trading days
- Trend score table using return, continuation rate, alpha, volatility, volume growth, and relative strength
- Volume Intelligence tab with buying pressure, selling pressure, accumulation score, market interest, liquidity rating, breakout confirmation, and overall conviction score
- Volume spike reaction study showing what happened after unusually high volume days
- Benchmark comparison against Nifty 50 when live data is fetched
- Threshold testing for 3%, 4%, 5%, and 6% streak definitions
- Event-level backtest and risk metrics for selected signal strategies
- Histograms and heatmap
- Methodology and limitations tabs
- Downloadable formatted Excel report with dashboard, methodology, trend analysis, volume intelligence, volume spike reactions, rankings, threshold tests, backtest metrics, signals, limitations, and raw price data
- Offline cached-data mode for presentation safety

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Methodology

Default signal rules:

- Bullish signal: 3-day return >= +5%
- Bearish signal: 3-day return <= -5%
- Optional filter: all 3 days must move in the same direction
- Forward performance: next 3 trading days and next 5 trading days
- Trend lookback: default 15 trading days, configurable from the sidebar
- Volume spike threshold: default 1.5x rolling average volume, configurable from the sidebar
- Benchmark: Nifty 50 index (`^NSEI`) when using live data

This is exploratory statistical analysis, not investment advice.

## Volume Intelligence Module

The volume module uses daily OHLCV data to infer participation and conviction. It does not claim exact institutional buying or selling volume, because that cannot be proven from daily candles alone.

Price-volume classification:

- Price up + volume up: accumulation
- Price up + volume down: weak rally
- Price down + volume up: distribution
- Price down + volume down: weak selling

Main outputs:

- Buying Pressure Index
- Selling Pressure Index
- Accumulation Score
- Market Interest Index
- Volume Spike Ratio and category
- Breakout Status
- Institutional Activity Proxy
- Liquidity Rating
- Volume Intelligence Score
- Combined Conviction Score

## Backtest Module

The app includes a simple research backtest based on signal-level forward returns.

Available strategies:

- Bullish continuation long
- Bearish bounce long
- Bearish continuation short
- Combined long signals

The module reports trade count, average return, win rate, cumulative return, maximum drawdown, Sharpe-like score, profit factor, and alpha versus Nifty.

This is an event-level research simulation. It does not fully model overlapping trades, margin, liquidity, taxes, or real execution constraints.

## Presentation Safety

Use **Fetch latest data** once before presenting. The app saves the fetched data to universe-specific cache files:

```text
data/latest_price_data_nifty50.csv
data/latest_price_data_nifty500.csv
```

During the presentation, select **Use cached data** if the internet is slow or unavailable.

## Upload Format

You can also upload your own CSV or Excel file with these columns:

```text
Date, Stock, Open, High, Low, Close, Volume
```
