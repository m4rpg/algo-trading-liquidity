"""Step 1: LOBSTER messages -> stock x interval panel (processed_data/).

    python code/build_panel.py            # 5-minute panel (main)
    python code/build_panel.py 60 600 900 # extra widths for robustness
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from at_liquidity.io import load_raw, ROOT            # noqa: E402
from at_liquidity.panel import stock_panel             # noqa: E402

OUT = ROOT / "processed_data"


def main(widths=(300,)):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    raw = load_raw()
    print(f"loaded {', '.join(raw)} in {time.time() - t0:.0f}s")
    info = {}
    for w in widths:
        panels = []
        for t, msg in raw.items():
            p, tr, tm = stock_panel(t, msg, width=w)
            panels.append(p)
            if w == 300:
                tr.assign(ticker=t).to_parquet(OUT / f"trades_{t}.parquet")
                info[t] = {"messages": int(msg["type"].isin([1, 2, 3, 4, 5]).sum()),
                           "executions": int(msg["type"].isin([4, 5]).sum()),
                           "trades": int(len(tr)),
                           "hidden_exec_share": float((msg["type"] == 5).sum() / msg["type"].isin([4, 5]).sum())}
            print(f"  width {w:>4}s  {t}: {len(p)} intervals, {len(tr):,} trades  ({time.time() - t0:.0f}s)")
        pd.concat(panels, ignore_index=True).to_csv(OUT / f"panel_{w}s.csv", index=False)
    if info:
        (ROOT / "results").mkdir(exist_ok=True)
        json.dump(info, open(ROOT / "results" / "data_info.json", "w"), indent=1)


if __name__ == "__main__":
    widths = tuple(int(a) for a in sys.argv[1:]) or (300,)
    main(widths)
