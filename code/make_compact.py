"""Optional step: original LOBSTER sample files -> raw_data/LOBSTER_<TICKER>_compact.csv.gz

The orderbook files are large (up to 170 MB) because they hold 10 levels.
The paper only needs the best bid and ask, so we keep the message file and
the first four orderbook columns. Same rows, same order.

    python code/make_compact.py path/to/LOBSTER_SampleFile_AAPL_2012-06-21_10 ...
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MSG = ["time", "type", "order_id", "size", "price", "direction"]
OB = ["ask1", "asksize1", "bid1", "bidsize1"]


def compact(folder):
    folder = Path(folder)
    msg_file = next(folder.glob("*_message_*.csv"))
    ob_file = next(folder.glob("*_orderbook_*.csv"))
    ticker = msg_file.name.split("_")[0]
    msg = pd.read_csv(msg_file, header=None, usecols=range(6), names=MSG)
    ob = pd.read_csv(ob_file, header=None, usecols=range(4), names=OB)
    assert len(msg) == len(ob), "message and orderbook files must have the same rows"
    out = ROOT / "raw_data" / f"LOBSTER_{ticker}_compact.csv.gz"
    out.parent.mkdir(exist_ok=True)
    pd.concat([msg, ob], axis=1).to_csv(out, index=False, float_format="%.9f")
    print(f"{ticker}: {len(msg):,} messages -> {out.name}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        compact(f)
