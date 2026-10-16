"""From LOBSTER messages to a table of trades.

One marketable order that walks the book is logged as several executions
with the same timestamp. We merge them into one trade, as TAQ would show it.
"""
import numpy as np
import pandas as pd

EXEC_TYPES = (4, 5)   # 4 = execution of a visible order, 5 = of a hidden order


def build_trades(msg, horizons=(300, 1800)):
    """Trades with sign, prevailing quotes and future midpoints.

    LOBSTER `direction` is the side of the *resting* limit order:
    +1 = a buy order was hit, so the trade was seller-initiated (q = -1).
    This is the true initiator, so we do not need the Lee-Ready algorithm.
    """
    ex = msg[msg["type"].isin(EXEC_TYPES)].copy()
    ex["q"] = -ex["direction"]
    ex["dollar"] = ex["price"] * ex["size"]
    # new trade when the timestamp or the side changes
    new = (ex["time"].diff() != 0) | (ex["q"].diff() != 0)
    ex["trade_id"] = new.cumsum()
    tr = ex.groupby("trade_id").agg(
        time=("time", "first"), q=("q", "first"),
        size=("size", "sum"), dollar=("dollar", "sum"),
        n_exec=("size", "size"), hidden=("type", lambda s: (s == 5).mean()),
        bid=("pre_bid1", "first"), ask=("pre_ask1", "first"),
        bidsize=("pre_bidsize1", "first"), asksize=("pre_asksize1", "first"),
        mid=("pre_mid", "first"),
    ).reset_index(drop=True)
    tr["price"] = tr["dollar"] / tr["size"]
    tr = tr[tr["mid"].notna() & (tr["ask"] > tr["bid"])].reset_index(drop=True)

    # midpoint h seconds after each trade = state after the last message <= t + h
    t_msg = msg["time"].to_numpy()
    mid_msg = msg["mid"].to_numpy()
    for h in horizons:
        idx = np.searchsorted(t_msg, tr["time"].to_numpy() + h, side="right") - 1
        fut = mid_msg[idx]
        fut[tr["time"].to_numpy() + h > t_msg[-1]] = np.nan   # beyond the close
        tr[f"mid_{h // 60}m"] = fut
    return tr


def trade_measures(tr):
    """Equations (1)-(4) of the paper, per trade, in basis points of the midpoint."""
    m, p, q = tr["mid"], tr["price"], tr["q"]
    out = pd.DataFrame(index=tr.index)
    out["qspread"] = (tr["ask"] - tr["bid"]) / 2 / m * 1e4                   # quoted half-spread
    out["qdepth"] = (tr["ask"] * tr["asksize"] + tr["bid"] * tr["bidsize"]) / 2 / 1000  # $1,000
    out["espread"] = q * (p - m) / m * 1e4                                   # eq. (1)
    for h in ("5m", "30m"):
        out[f"rspread_{h}"] = q * (p - tr[f"mid_{h}"]) / m * 1e4             # eq. (2)
        out[f"adv_sel_{h}"] = q * (tr[f"mid_{h}"] - m) / m * 1e4             # eq. (3)
    return out
