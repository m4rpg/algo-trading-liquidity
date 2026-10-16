"""Step 0 / sanity checks: run after build_panel.py. Writes results/checks.json.

1. Identity (4): effective = realized + adverse selection, trade by trade.
2. Trade signs: buyer-initiated trades should print at or above the midpoint.
3. Hasbrouck VAR code recovers a known price impact on simulated data.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from at_liquidity.io import ROOT                       # noqa: E402
from at_liquidity.trades import trade_measures         # noqa: E402
from at_liquidity.hasbrouck import decompose           # noqa: E402


def simulated_var(lam=2e-4, sigma_u=1e-4, n=20_000, seed=0):
    """Efficient price m_t = m_{t-1} + lam*e_q + u_t, quotes = m, persistent order flow."""
    rng = np.random.default_rng(seed)
    q = np.empty(n)
    q[0] = 1
    for t in range(1, n):                     # order flow persists: P(same sign) = 0.7
        q[t] = q[t - 1] if rng.random() < 0.7 else -q[t - 1]
    surprise = q - 0.4 * np.r_[0, q[:-1]]     # E[q_t | q_{t-1}] = 0.4 q_{t-1}
    r = lam * surprise + sigma_u * rng.standard_normal(n)
    out = decompose(r, q)
    var_q = np.var(surprise)
    true_share = lam ** 2 * var_q / (lam ** 2 * var_q + sigma_u ** 2)
    return {"true_impulse_bps": lam * 1e4, "estimated_impulse_bps": out["impulse_response"],
            "true_trade_share": true_share, "estimated_trade_share": out["share_tradecorr"]}


def main():
    proc = ROOT / "processed_data"
    res = {}
    for f in sorted(proc.glob("trades_*.parquet")):
        tr = pd.read_parquet(f)
        t = f.stem.split("_")[1]
        tm = trade_measures(tr)
        ok = tm[["rspread_5m", "adv_sel_5m"]].notna().all(axis=1)
        gap = (tm.loc[ok, "espread"] - tm.loc[ok, "rspread_5m"] - tm.loc[ok, "adv_sel_5m"]).abs().max()
        signed = tr["q"] * (tr["price"] - tr["mid"])
        res[t] = {"trades": int(len(tr)), "identity_max_gap_bps": float(gap),
                  "share_on_correct_side": float((signed >= -1e-9).mean()),
                  "share_inside_spread": float(((tr["price"] > tr["bid"]) & (tr["price"] < tr["ask"])).mean())}
    res["simulated_var"] = simulated_var()
    json.dump(res, open(ROOT / "results" / "checks.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert all(v["identity_max_gap_bps"] < 1e-8 for k, v in res.items() if k != "simulated_var")
    s = res["simulated_var"]
    assert abs(s["estimated_impulse_bps"] / s["true_impulse_bps"] - 1) < 0.05
    print("all checks passed")


if __name__ == "__main__":
    main()
