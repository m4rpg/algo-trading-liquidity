"""Loading LOBSTER data.

We keep only what the paper needs: every message (to count message traffic)
and the best bid / ask after every message (to measure spreads).
The compact files are made from the original LOBSTER sample files, see README.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
OPEN, CLOSE = 34200.0, 57600.0          # 9:30 and 16:00, seconds after midnight
PRICE_SCALE = 10_000                    # LOBSTER prices are in 1/10000 dollars
DUMMY = 9_999_999_999                   # LOBSTER fills an empty side with +-9999999999

COLS = ["time", "type", "order_id", "size", "price", "direction",
        "ask1", "asksize1", "bid1", "bidsize1"]

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "raw_data"


def _read_compact(path):
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        df.insert(0, "ticker", path.name.split("_")[1])
    return df


def load_raw(raw_dir=RAW):
    """All compact files in raw_data/ -> {ticker: DataFrame}."""
    frames = [_read_compact(p) for p in sorted(Path(raw_dir).glob("LOBSTER_*_compact.csv.gz"))]
    if not frames:
        raise FileNotFoundError(f"no LOBSTER_*_compact.csv.gz in {raw_dir}, see README")
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for t, g in df.groupby("ticker", sort=True):
        out[t] = clean(g.drop(columns="ticker").reset_index(drop=True))
    return out


def clean(df):
    """Dollar prices, drop trading halts, mark empty book sides as NaN."""
    df = df[df["type"] != 7].copy()                        # 7 = trading halt
    for c in ["price", "ask1", "bid1"]:
        df[c] = df[c].astype(float)
    df.loc[df["ask1"].abs() >= DUMMY, "ask1"] = np.nan
    df.loc[df["bid1"].abs() >= DUMMY, "bid1"] = np.nan
    for c in ["price", "ask1", "bid1"]:
        df[c] = df[c] / PRICE_SCALE
    df["mid"] = (df["ask1"] + df["bid1"]) / 2
    # the book columns show the state AFTER the message; the quote a message
    # (or a trade) meets is the state before it
    for c in ["ask1", "bid1", "asksize1", "bidsize1", "mid"]:
        df["pre_" + c] = df[c].shift(1)
    return df.reset_index(drop=True)
