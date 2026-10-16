"""Step 2: the paper's regressions on our stock x interval panel -> results/.

Main specification: 10-minute intervals, the paper's AT proxy (algo_trad),
equation (7) with stock and interval fixed effects and the paper's controls.

Table I    summary statistics by stock
Table III  AT -> spreads, depth, spread decomposition
Table IV   AT -> turnover, trades, trade size
Table V    AT -> Hasbrouck impulse response and variance components
IV         2SLS with the Hasbrouck & Saar (2013) instrument (Table II = first stage)
Extension  a direct measure of algorithms (fleeting orders); interval width
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from at_liquidity.io import ROOT                                    # noqa: E402
from at_liquidity.regress import fit, first_stage, winsorize         # noqa: E402

PROC, RES = ROOT / "processed_data", ROOT / "results"
MAIN_WIDTH = 600          # seconds; every 10-minute interval has >= 60 trades for the VAR

GROUPS = {"All": ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"],
          "High-price": ["AAPL", "AMZN", "GOOG"],     # > $200: spread spans many ticks
          "Low-price": ["INTC", "MSFT"]}              # ~ $30: spread is mostly one tick
CONTROLS = ["turnover", "volatility", "inv_price"]     # ln mkt cap = ln price + const within one day

# (dependent variable, uses controls?, table) -- as in the paper
SPEC = [("qspread", True, "III"), ("qdepth", True, "III"), ("espread", True, "III"),
        ("rspread_5m", True, "III"), ("adv_sel_5m", True, "III"),
        ("rspread_30m", True, "III"), ("adv_sel_30m", True, "III"),
        ("turnover", False, "IV"), ("trades", False, "IV"), ("trade_size", False, "IV"),
        ("impulse_response", True, "V"), ("sd_tradecorr", False, "V"), ("sd_nontradecorr", False, "V")]

WINSOR = [s[0] for s in SPEC] + ["algo_trad", "messages", "volatility", "fleet_100ms",
                                  "fleet_1000ms", "cancel_to_trade"]

# the paper's coefficient on algo_trad for quintile 1 (Tables III-V);
# all five of our stocks are among the largest US companies, i.e. quintile 1
PAPER_Q1 = {"qspread": (-0.53, -3.23), "qdepth": (-3.49, -2.51), "espread": (-0.18, -2.67),
            "rspread_5m": (0.35, 3.52), "adv_sel_5m": (-0.53, -3.56),
            "rspread_30m": (0.33, 2.82), "adv_sel_30m": (-0.51, -3.43),
            "turnover": (0.04, 1.02), "trades": (0.58, 2.60), "trade_size": (-2.04, -4.64),
            "impulse_response": (-0.54, -3.50), "sd_tradecorr": (-0.22, -2.62),
            "sd_nontradecorr": (0.12, 2.48)}


def loo_instrument(p, col):
    """Mean of the OTHER stocks' within-stock z-score of `col` in the same interval."""
    z = p.groupby("ticker")[col].transform(lambda s: (s - s.mean()) / s.std())
    tot = z.groupby(p["bin"]).transform("sum")
    cnt = z.notna().groupby(p["bin"]).transform("sum")
    return (tot - z) / (cnt - 1)


def prepare(panel):
    p = panel.copy()
    for c in WINSOR:
        if c in p:
            p[c] = winsorize(p[c])
    p["halfhour"] = ((p["clock"] - 9.5) // 0.5).astype(int)
    p["log_messages"] = np.log(p["messages"])
    p["inst_algo"] = loo_instrument(p, "algo_trad")
    p["inst_msg"] = loo_instrument(p, "log_messages")
    p["inst_fleet"] = loo_instrument(p, "fleet_100ms")
    return p


def within_sd(d, c):
    return (d[c] - d.groupby("ticker")[c].transform("mean")).std()


def ols_table(p, at="algo_trad"):
    rows = []
    for g, tick in GROUPS.items():
        d = p[p["ticker"].isin(tick)]
        sd = within_sd(d, at)
        for y, ctrl, tab in SPEC:
            r = fit(d, y, at, CONTROLS if ctrl else (), None, fe_time="bin")
            pc, pt = PAPER_Q1[y]
            rows.append({"table": tab, "group": g, "dep": y, "at": at,
                         "coef": r["coef"], "t": r["t"], "p": r["p"], "n": r["n"],
                         "mean_y": d[y].mean(), "within_sd_at": sd,
                         "effect_1sd": r["coef"] * sd,
                         "effect_1sd_pct": r["coef"] * sd / d[y].mean() * 100,
                         "paper_coef": pc, "paper_t": pt,
                         "same_sign": bool(np.sign(r["coef"]) == np.sign(pc))})
    return pd.DataFrame(rows)


def iv_table(p, at, inst):
    rows = []
    for g, tick in GROUPS.items():
        d = p[p["ticker"].isin(tick)]
        fs = first_stage(d, at, inst, CONTROLS)
        for y, ctrl, tab in SPEC:
            if tab == "IV":
                continue
            r = fit(d, y, at, CONTROLS if ctrl else (), inst, fe_time="halfhour")
            rows.append({"group": g, "dep": y, "at": at, "instrument": inst,
                         "coef": r["coef"], "t": r["t"], "p": r["p"], "n": r["n"],
                         "first_stage_coef": fs["coef"], "first_stage_t": fs["t"], "F": fs["F"],
                         "paper_coef": PAPER_Q1[y][0],
                         "same_sign": bool(np.sign(r["coef"]) == np.sign(PAPER_Q1[y][0]))})
    return pd.DataFrame(rows)


def main():
    RES.mkdir(exist_ok=True)
    p = prepare(pd.read_csv(PROC / f"panel_{MAIN_WIDTH}s.csv"))
    p.to_csv(PROC / "panel_main_prepared.csv", index=False)
    out = {"main_width_s": MAIN_WIDTH, "n_obs": len(p), "n_intervals": int(p["bin"].nunique())}

    # Table I -----------------------------------------------------------------
    t1_cols = ["qspread", "qdepth", "espread", "rspread_5m", "adv_sel_5m", "rspread_30m",
               "adv_sel_30m", "messages", "algo_trad", "dollar_volume", "trades", "volatility",
               "price", "trade_size", "cancel_to_trade", "fleet_100ms", "fleet_1000ms",
               "impulse_response", "sd_tradecorr", "sd_nontradecorr", "share_tradecorr"]
    t1 = p.groupby("ticker")[t1_cols].mean().T
    t1["All"] = p[t1_cols].mean()
    t1["within_sd"] = [within_sd(p, c) for c in t1_cols]
    t1.to_csv(RES / "table1_summary.csv")
    out["table1"] = t1.round(4).to_dict()

    # Tables III-V: OLS with two-way fixed effects ----------------------------
    ols = ols_table(p, "algo_trad")
    ols_msg = ols_table(p, "log_messages")
    ols.to_csv(RES / "ols_algo_trad.csv", index=False)
    ols_msg.to_csv(RES / "ols_log_messages.csv", index=False)
    out["ols_algo_trad"] = ols.to_dict(orient="records")
    out["ols_log_messages"] = ols_msg.to_dict(orient="records")
    out["sign_match"] = {
        f"{m}_{g}": {"same": int(t[t.group == g].same_sign.sum()), "of": int((t.group == g).sum()),
                     "same_and_sig": int((t[t.group == g].same_sign & (t[t.group == g].p < 0.05)).sum()),
                     "opposite_and_sig": int((~t[t.group == g].same_sign & (t[t.group == g].p < 0.05)).sum())}
        for m, t in [("algo_trad", ols), ("log_messages", ols_msg)] for g in GROUPS}

    # IV (Table II = first stage) --------------------------------------------
    iv1 = iv_table(p, "algo_trad", "inst_algo")
    iv2 = iv_table(p, "log_messages", "inst_msg")
    iv = pd.concat([iv1, iv2], ignore_index=True)
    iv.to_csv(RES / "iv_2sls.csv", index=False)
    out["iv"] = iv.to_dict(orient="records")
    t2 = []
    for g, tick in GROUPS.items():
        d = p[p["ticker"].isin(tick)]
        for y, inst in [("algo_trad", "inst_algo"), ("messages", "inst_algo"),
                        ("log_messages", "inst_msg"), ("turnover", "inst_algo"),
                        ("volatility", "inst_algo"), ("volatility", "inst_msg")]:
            r = first_stage(d, y, inst, CONTROLS if y in ("algo_trad", "log_messages") else ())
            t2.append({"group": g, "dep": y, "instrument": inst, "coef": r["coef"],
                       "t": r["t"], "F": r["F"], "n": r["n"]})
    pd.DataFrame(t2).to_csv(RES / "table2_first_stage.csv", index=False)
    out["table2"] = t2

    # Extension 1: does the paper's proxy track real algorithms? ---------------
    cols = ["algo_trad", "messages", "cancel_to_trade", "fleet_100ms", "fleet_1000ms", "turnover", "volatility"]
    for name, df in [("main", p), ("5min", prepare(pd.read_csv(PROC / "panel_300s.csv")))]:
        w = df[cols].copy()
        for c in cols:
            w[c] = df[c] - df.groupby("ticker")[c].transform("mean")
        corr = w.corr()
        corr.to_csv(RES / f"at_proxy_correlations_{name}.csv")
        out[f"at_proxy_corr_{name}"] = corr.round(3).to_dict()
        # cross-section: which stock looks "most algorithmic" by each measure
    out["at_rank"] = p.groupby("ticker")[["algo_trad", "messages", "cancel_to_trade", "fleet_100ms"]].mean().round(4).to_dict()
    ols_fl = ols_table(p, "fleet_100ms")
    ols_fl.to_csv(RES / "ols_fleeting.csv", index=False)
    out["ols_fleeting"] = ols_fl.to_dict(orient="records")

    # Extension 2: interval width ---------------------------------------------
    rob = []
    for wdt in [60, 300, 600, 900]:
        f = PROC / f"panel_{wdt}s.csv"
        if not f.exists():
            continue
        q = prepare(pd.read_csv(f))
        for g in ["All", "High-price"]:
            d = q[q["ticker"].isin(GROUPS[g])]
            for y in ["qspread", "espread", "rspread_5m", "adv_sel_5m", "impulse_response"]:
                for at in ["algo_trad", "log_messages"]:
                    r = fit(d, y, at, CONTROLS, None, fe_time="bin")
                    rob.append({"width_s": wdt, "group": g, "dep": y, "at": at,
                                "coef": r["coef"], "t": r["t"], "n": r["n"],
                                "same_sign": bool(np.sign(r["coef"]) == np.sign(PAPER_Q1[y][0]))})
    rob = pd.DataFrame(rob)
    rob.to_csv(RES / "robustness_width.csv", index=False)
    out["robustness_width"] = rob.to_dict(orient="records")

    json.dump(out, open(RES / "results.json", "w"), indent=1, default=float)

    pd.set_option("display.width", 200)
    print(ols[["group", "dep", "coef", "t", "effect_1sd_pct", "paper_coef", "same_sign"]].round(3).to_string())
    print(json.dumps(out["sign_match"], indent=0))
    print(iv[["group", "dep", "at", "coef", "t", "F"]].round(3).to_string())
    print(pd.DataFrame(t2).round(3).to_string())


if __name__ == "__main__":
    main()
