"""Direct measures of algorithmic activity (our extension).

The paper cannot see algorithms and uses message traffic per $100 of volume
as a proxy. LOBSTER has order ids and nanosecond timestamps, so we can see
behaviour that no human can produce: limit orders cancelled within
milliseconds of being submitted ("fleeting orders", Hasbrouck & Saar 2009).
"""
import numpy as np
import pandas as pd


def order_lifetimes(msg):
    """One row per submitted limit order: submission time and how it ended."""
    sub = msg.loc[msg["type"] == 1, ["time", "order_id", "size"]]
    sub = sub.drop_duplicates("order_id").set_index("order_id")
    dele = msg.loc[msg["type"] == 3].groupby("order_id")["time"].first()
    exe = msg.loc[msg["type"] == 4].groupby("order_id")["time"].first()
    out = sub.rename(columns={"time": "t_sub"})
    out["t_del"] = dele.reindex(out.index)
    out["t_exec"] = exe.reindex(out.index)
    out["life_del"] = out["t_del"] - out["t_sub"]
    # fully cancelled without any execution before the cancel
    out["cancel_no_fill"] = out["t_del"].notna() & ~(out["t_exec"] < out["t_del"])
    return out.reset_index()


def fleeting_flags(life, cutoffs=(0.1, 1.0)):
    """Flags for orders cancelled (unfilled) within each cutoff, in seconds."""
    out = life[["t_sub"]].copy()
    for c in cutoffs:
        out[f"fleet_{int(c * 1000)}ms"] = (life["cancel_no_fill"] & (life["life_del"] <= c)).astype(float)
    return out
