# Research Seminar, Group 6 — Does Algorithmic Trading Improve Liquidity?

An adaptation of Hendershott, Jones & Menkveld (2011), *Journal of Finance* 66(1), 1–33,
on NASDAQ order-book data (LOBSTER): five stocks, 21 June 2012.

The paper uses TAQ and NYSE System Order Data for 1,082 stocks in 2002–2003 and the
staggered NYSE autoquote rollout as an instrument. Those data are not public, so we
rebuild every measure of the paper from raw messages and ask whether the same
relations show up nine years later on another exchange.

## Data

Download from https://huggingface.co/datasets/y7pan/lobster-data the folders
`LOBSTER_SampleFile_<TICKER>_2012-06-21_10` for AAPL, AMZN, GOOG, INTC, MSFT and
shrink them to the columns we use (all messages + best bid/ask):

```
python code/make_compact.py path/to/LOBSTER_SampleFile_AAPL_2012-06-21_10 path/to/...AMZN... 
```

This writes `raw_data/LOBSTER_<TICKER>_compact.csv.gz` (25 MB in total instead of 600 MB).

## Run

```
pip install -r requirements.txt
python code/build_panel.py 300 60 600 900   # messages -> trades -> stock x interval panels (~1 min)
python code/checks.py                       # identity (4), trade signs, VAR on simulated data
python code/run_regressions.py              # Tables I-V, IV, extensions -> results/
python code/make_figures.py                 # figures/, results/summary.json
python code/make_notebook.py                # AT_Liquidity_Walkthrough.ipynb
```

## Structure

```
code/at_liquidity/io.py         loading, empty book sides, quote before each message
code/at_liquidity/trades.py     executions -> trades, eq. (1)-(4): effective, realized spread, adverse selection
code/at_liquidity/hasbrouck.py  Hasbrouck VAR, eq. (8)-(11): price impact, trade-(un)correlated variance
code/at_liquidity/fleeting.py   order lifetimes: orders cancelled within 100 ms (our extension)
code/at_liquidity/panel.py      stock x interval panel with all Table I variables
code/at_liquidity/regress.py    two-way FE OLS and 2SLS with Driscoll-Kraay standard errors
AT_Liquidity_Walkthrough.ipynb  data walkthrough and results
results/                        all tables (csv, json); figures/ all charts
```

## Method in one paragraph

Panel of 5 stocks × 39 ten-minute intervals (the paper: stocks × days). AT proxy as in the
paper: `algo_trad = −(dollar volume / 100) / messages`. Liquidity measures are share-weighted
quoted, effective and realized half-spreads, adverse selection at 5 and 30 minutes, quoted depth,
and the Hasbrouck VAR (10 lags) per stock-interval. Regression (7): stock and interval fixed effects,
controls turnover, volatility (Parkinson), 1/price. Standard errors: Driscoll–Kraay.
Groups: high-price stocks (AAPL, AMZN, GOOG; spread spans many ticks) and low-price stocks
(INTC, MSFT; spread mostly one tick). All five are quintile-1 (largest) stocks in the paper's terms.

## Results

High-price stocks, 10-minute intervals, OLS with two-way fixed effects (t-statistics):

| Outcome | Paper Q1 (IV) | Ours, `algo_trad` | Ours, log messages |
|---|---|---|---|
| Quoted half-spread | −0.53 (−3.2) | −0.010 (−1.7) | −0.34 (−2.2) |
| Effective half-spread | −0.18 (−2.7) | −0.014 (−2.9) | −0.32 (−2.3) |
| Realized spread, 5 min | +0.35 (+3.5) | +0.021 (+1.1) | +3.22 (+4.2) |
| Adverse selection, 5 min | −0.53 (−3.6) | −0.032 (−1.6) | −3.52 (−4.6) |
| Price impact (VAR) | −0.54 (−3.5) | −0.008 (−1.1) | +0.17 (+0.4) |
| Trade size | −2.04 (−4.6) | −0.41 (−2.9) | +4.74 (+1.2) |

* Same sign as the paper: 9 of 13 outcomes (`algo_trad`), 10 of 13 (log messages; 5 significant, none significant with the opposite sign).
* The 30-minute decomposition and the trade count do not replicate.
* IV with "AT in the other stocks" (Hasbrouck & Saar 2013) fails: first-stage F = 3.1–6.4, and the
  instrument predicts turnover (F = 20.9) — the exclusion restriction does not hold intraday.
* Extension: within a stock, `algo_trad` correlates −0.68 with turnover but only −0.08 with the share of
  orders cancelled within 100 ms (32% of all 1.03 m submitted orders). Intraday, the paper's proxy mostly
  measures volume; message counts track the visible algorithms (r = 0.45).
