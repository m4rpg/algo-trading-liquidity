"""Stock x interval panel: the paper's stock x day panel, moved to intraday time.

Every variable follows the paper's Table I definition; only the time unit
changes (a 5-minute interval instead of a trading day).
"""
import numpy as np
import pandas as pd

from .io import OPEN, CLOSE
from .trades import build_trades, trade_measures
from .hasbrouck import decompose
from .fleeting import order_lifetimes, fleeting_flags

MSG_TYPES = (1, 2, 3, 4, 5)   # submissions, partial cancels, deletions, executions


def _bin(t, width):
    b = ((t - OPEN) // width).astype(int)
    return b.clip(0, int((CLOSE - OPEN) // width) - 1)


def _wavg(x, w):
    ok = x.notna()
    return (x[ok] * w[ok]).sum() / w[ok].sum() if w[ok].sum() > 0 else np.nan


def stock_panel(ticker, msg, width=300, min_var_trades=60):
    msg = msg.copy()
    msg["bin"] = _bin(msg["time"], width)
    minutes = width / 60

    tr = build_trades(msg)
    tm = trade_measures(tr)
    tm["bin"] = _bin(tr["time"], width)
    tm["shares"] = tr["size"]

    # share-weighted liquidity measures, as in the paper
    liq_cols = ["qspread", "qdepth", "espread", "rspread_5m", "adv_sel_5m",
                "rspread_30m", "adv_sel_30m"]
    liq = tm.groupby("bin").apply(
        lambda g: pd.Series({c: _wavg(g[c], g["shares"]) for c in liq_cols}),
        include_groups=False)

    # activity
    m = msg[msg["type"].isin(MSG_TYPES)]
    act = pd.DataFrame({
        "messages": m.groupby("bin").size() / minutes,                 # per minute
        "submissions": (m["type"] == 1).groupby(m["bin"]).sum() / minutes,
        "cancels": m["type"].isin((2, 3)).groupby(m["bin"]).sum() / minutes,
    })
    vol = tr.assign(bin=_bin(tr["time"], width)).groupby("bin").agg(
        dollar_volume=("dollar", "sum"), share_volume=("size", "sum"),
        n_trades=("size", "size"), hidden_share=("hidden", "mean"))
    mids = msg.groupby("bin")["mid"].agg(["max", "min", "mean", "first", "last"])

    p = act.join(vol).join(liq).join(mids)
    p["algo_trad"] = -(p["dollar_volume"] / 100) / (p["messages"] * minutes)  # -$100 per message
    p["trades"] = p["n_trades"] / minutes
    p["trade_size"] = p["dollar_volume"] / p["n_trades"] / 1000           # $1,000
    p["turnover"] = p["share_volume"] / p["share_volume"].mean()          # relative to the stock's day
    p["volatility"] = np.log(p["max"] / p["min"]) / np.sqrt(4 * np.log(2)) * 100   # Parkinson, %
    p["price"] = p["mean"]
    p["inv_price"] = 1 / p["mean"]
    p["ret"] = np.log(p["last"] / p["first"]) * 100
    p["cancel_to_trade"] = p["cancels"] / p["trades"]

    # direct algorithm measures
    fl = fleeting_flags(order_lifetimes(msg))
    fl["bin"] = _bin(fl["t_sub"], width)
    p = p.join(fl.groupby("bin")[["fleet_100ms", "fleet_1000ms"]].mean())

    # Hasbrouck VAR per interval (needs enough trades)
    r = np.log(tr["mid"]).diff().shift(-1)          # mid before t -> mid before t+1
    tr2 = tr.assign(r=r, bin=_bin(tr["time"], width)).dropna(subset=["r"])
    rows = {}
    for b, g in tr2.groupby("bin"):
        if len(g) >= min_var_trades:
            rows[b] = decompose(g["r"].to_numpy(), g["q"].to_numpy())
    p = p.join(pd.DataFrame(rows).T)

    p = p.drop(columns=["max", "min", "mean", "first", "last"])
    p.insert(0, "ticker", ticker)
    p.index.name = "bin"
    p = p.reset_index()
    p["clock"] = (OPEN + p["bin"] * width) / 3600       # hours, start of interval
    return p, tr, tm
