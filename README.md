# Does Algorithmic Trading Improve Liquidity?

Research Seminar "Data Analysis in Finance", HSE, Group 6.

This project replicates the analysis of Hendershott, Jones and Menkveld (2011), "Does Algorithmic Trading Improve Liquidity?", Journal of Finance 66(1), 1-33. The original study uses TAQ and NYSE System Order Data for 2002-2003, which are not publicly available. We therefore use LOBSTER data, which contain all NASDAQ messages for AAPL, AMZN, GOOG, INTC and MSFT on 21 June 2012.

## Project overview

1. We reconstruct the measures used in the paper from raw messages: the algorithmic trading (AT) proxy, quoted, effective and realized spreads, adverse selection at 5 and 30 minutes, quoted depth, and the Hasbrouck (1991) VAR.
2. Since our data cover a single day, the unit of observation is a stock-interval (5 stocks x 39 ten-minute intervals) instead of a stock-day.
3. We estimate the paper's main specification (equation 7) with stock and interval fixed effects and Driscoll-Kraay standard errors.
4. As the autoquote event is not available, we test the instrument proposed by Hasbrouck and Saar (2013).
5. As an extension, we measure algorithmic activity directly through orders cancelled within 100 ms and compare it with the paper's proxy.

## Data

The LOBSTER sample files are available at https://huggingface.co/datasets/y7pan/lobster-data (folders `LOBSTER_SampleFile_<TICKER>_2012-06-21_10`). The original files take about 600 MB, so we keep only the required columns (all messages and the best bid and ask). The reduced files are stored in `raw_data/` and can be recreated with:

```
python code/make_compact.py path/to/LOBSTER_SampleFile_AAPL_2012-06-21_10
```

## Reproduction

```
pip install -r requirements.txt
python code/build_panel.py 300 60 600 900
python code/checks.py
python code/run_regressions.py
python code/make_figures.py
```

The full pipeline runs in about two minutes. The results are summarised in `AT_Liquidity_Walkthrough.ipynb`.

## Repository structure

```
code/at_liquidity/io.py          data loading
code/at_liquidity/trades.py      trades and spread measures (eq. 1-4)
code/at_liquidity/hasbrouck.py   Hasbrouck VAR (eq. 8-11)
code/at_liquidity/fleeting.py    order lifetimes and fleeting orders
code/at_liquidity/panel.py       stock x interval panel
code/at_liquidity/regress.py     OLS and 2SLS with fixed effects
code/build_panel.py              step 1: panel construction
code/checks.py                   consistency checks
code/run_regressions.py          step 2: regression tables
code/make_figures.py             step 3: figures
results/                         output tables (csv, json)
figures/                         charts
```

## Main results

High-price stocks (AAPL, AMZN, GOOG), 10-minute intervals, t-statistics in parentheses:

| | Paper, quintile 1 | Ours, algo_trad | Ours, log messages |
|---|---|---|---|
| Quoted half-spread | -0.53 (-3.2) | -0.010 (-1.7) | -0.34 (-2.2) |
| Effective half-spread | -0.18 (-2.7) | -0.014 (-2.9) | -0.32 (-2.3) |
| Realized spread, 5 min | 0.35 (3.5) | 0.021 (1.1) | 3.22 (4.2) |
| Adverse selection, 5 min | -0.53 (-3.6) | -0.032 (-1.6) | -3.52 (-4.6) |

- With the paper's proxy (algo_trad), 9 of 13 coefficients have the same sign as in the paper. With log messages, 10 of 13 coefficients match, 5 of them are statistically significant, and none is significant with the opposite sign.
- The 30-minute spread decomposition and the number of trades are not consistent with the paper.
- The instrument based on AT in other stocks (Hasbrouck and Saar, 2013) is weak, with a first-stage F statistic between 3.1 and 6.4. It also predicts turnover (F = 20.9), which suggests that the exclusion restriction does not hold for intraday data.
- Extension: 32% of the 1.03 million submitted orders are cancelled within 100 ms. Within a stock, algo_trad has a correlation of -0.68 with turnover but only -0.08 with the share of these orders. This indicates that, at the intraday level, the paper's proxy mainly reflects trading volume, while message counts are more closely related to algorithmic activity (r = 0.45).
