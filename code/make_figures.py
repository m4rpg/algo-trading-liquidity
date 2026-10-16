"""Step 3: figures/ and results/summary.json from processed_data/ and results/."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from at_liquidity.io import ROOT, load_raw                 # noqa: E402
from at_liquidity.fleeting import order_lifetimes          # noqa: E402

RES, FIG, PROC = ROOT / "results", ROOT / "figures", ROOT / "processed_data"

NAVY = "#0D2D6A"
LIME = "#D4F500"
LAVENDER = "#C7B0EA"
GRAY = "#8A8FA3"
INK = "#3A4260"
STOCK_COLORS = {"AAPL": "#2A4F9E", "AMZN": "#9B7FD4", "GOOG": "#7E8C00",
                "INTC": "#3D9BC7", "MSFT": "#D9773A"}   # checked with the dataviz CVD validator
LABELS = {"qspread": "Quoted half-spread", "qdepth": "Quoted depth", "espread": "Effective half-spread",
          "rspread_5m": "Realized spread, 5 min", "adv_sel_5m": "Adverse selection, 5 min",
          "rspread_30m": "Realized spread, 30 min", "adv_sel_30m": "Adverse selection, 30 min",
          "turnover": "Turnover", "trades": "Trades per minute", "trade_size": "Trade size",
          "impulse_response": "Price impact (VAR)", "sd_tradecorr": "Trade-correlated volatility",
          "sd_nontradecorr": "Trade-uncorrelated volatility"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B8BCCB", "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.titlecolor": NAVY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#E6E8EF", "figure.dpi": 200, "savefig.bbox": "tight",
    "lines.linewidth": 2,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)


def intraday(p):
    for name, cols, ylabels, titles in [
        ("intraday_activity.png", ["messages", "algo_trad"],
         ["Messages per minute (log scale)", "algo_trad: −$100 of volume per message"],
         ["Message traffic", "The paper’s AT proxy"]),
        ("intraday_spreads.png", ["qspread", "espread"],
         ["Quoted half-spread, bps", "Effective half-spread, bps"],
         ["Quoted half-spread", "Effective half-spread"])]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, c, yl, tt in zip(axes, cols, ylabels, titles):
            for t, g in p.groupby("ticker"):
                ax.plot(g["clock"], g[c], color=STOCK_COLORS[t], label=t, lw=1.8)
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(yl)
            ax.set_title(tt)
            ax.set_xticks(range(10, 17))
            if c == "messages":
                ax.set_yscale("log")
        axes[0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(1.1, -0.18))
        save(fig, name)


def decomposition(t1):
    names = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT", "All"]
    paper = {"espread": 3.63, "rspread_5m": 1.21, "adv_sel_5m": 2.42}
    x = np.arange(len(names) + 1)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.26
    for k, (c, col, lab) in enumerate([("espread", NAVY, "Effective half-spread"),
                                       ("rspread_5m", LAVENDER, "Realized spread (liquidity supplier revenue)"),
                                       ("adv_sel_5m", "#A9BC00", "Adverse selection (price impact)")]):
        vals = [t1.loc[c, n] for n in names] + [paper[c]]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.08 if v >= 0 else -0.08), f"{v:.1f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color="#B8BCCB", lw=1)
    ax.set_xticks(x, names[:-1] + ["All 5\n(2012)", "NYSE Q1\n2003, paper"])
    ax.axvline(len(names) - 0.5, color=GRAY, lw=1, ls="--")
    ax.set_ylabel("Basis points of the midpoint")
    ax.set_title("Effective spread = realized spread + adverse selection (eq. 4)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    save(fig, "spread_decomposition.png")


def coef_comparison():
    a = pd.read_csv(RES / "ols_algo_trad.csv")
    m = pd.read_csv(RES / "ols_log_messages.csv")
    g = "High-price"
    a, m = a[a.group == g].set_index("dep"), m[m.group == g].set_index("dep")
    order = ["qspread", "qdepth", "espread", "rspread_5m", "adv_sel_5m", "rspread_30m", "adv_sel_30m",
             "impulse_response", "sd_tradecorr", "sd_nontradecorr", "trade_size", "trades", "turnover"]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.axvspan(-1.96, 1.96, color="#F3F4F8", zorder=0)
    ax.axvline(0, color="#B8BCCB", lw=1)
    ax.scatter(a.loc[order, "paper_t"], y, marker="D", s=70, color=NAVY, label="Paper, NYSE quintile 1 (2003)", zorder=3)
    ax.scatter(a.loc[order, "t"], y + 0.18, s=70, color="#9B7FD4", label="Ours: paper’s proxy (algo_trad)", zorder=3,
               edgecolor="white", lw=1.5)
    ax.scatter(m.loc[order, "t"], y - 0.18, s=70, color="#A9BC00", label="Ours: log messages (paper’s fn. 9)", zorder=3,
               edgecolor="white", lw=1.5)
    ax.set_yticks(y, [LABELS[o] for o in order], fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlabel("t-statistic of the AT coefficient (grey band: not significant at 5%)")
    ax.set_title("t-statistics: paper vs ours, high-price stocks", fontsize=15)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.3, -0.11), ncol=1, fontsize=12.5)
    ax.grid(axis="y", visible=False)
    save(fig, "coef_comparison.png")


def first_stage():
    t2 = pd.read_csv(RES / "table2_first_stage.csv")
    rows = [("Paper:\nautoquote → AT\n(Tables III, IV)", 7.32, NAVY)]
    for g in ["All", "High-price", "Low-price"]:
        r = t2[(t2.group == g) & (t2.dep == "algo_trad")].iloc[0]
        rows.append((f"Ours, {g}:\nothers’ AT\n→ own AT", r.F, LAVENDER))
    for g in ["All"]:
        r = t2[(t2.group == g) & (t2.dep == "turnover")].iloc[0]
        rows.append((f"Placebo:\nothers’ AT\n→ turnover", r.F, "#A9BC00"))
        r = t2[(t2.group == g) & (t2.dep == "volatility") & (t2.instrument == "inst_msg")].iloc[0]
        rows.append((f"Placebo:\nothers’ messages\n→ volatility", r.F, "#A9BC00"))
    fig, ax = plt.subplots(figsize=(12, 4.4))
    x = np.arange(len(rows))
    bars = ax.bar(x, [r[1] for r in rows], 0.6, color=[r[2] for r in rows])
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.5, f"{r[1]:.1f}", ha="center", color=INK, fontsize=11)
    ax.axhline(10, color="#C0392B", lw=1.5, ls="--")
    ax.text(len(rows) - 0.55, 10.4, "rule of thumb F = 10", color="#C0392B", ha="right", fontsize=10)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=9)
    ax.set_ylabel("First-stage F statistic")
    ax.set_title("Our instrument is weak for AT, but strong for things it should not move")
    ax.grid(axis="x", visible=False)
    save(fig, "first_stage.png")


def proxy_validity(p5):
    w = p5.copy()
    for c in ["algo_trad", "turnover", "fleet_100ms"]:
        w[c] = w[c] - w.groupby("ticker")[c].transform("mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, c, lab in [(axes[0], "turnover", "Turnover (deviation from stock mean)"),
                       (axes[1], "fleet_100ms", "Share of orders cancelled < 100 ms (deviation)")]:
        for t, g in w.groupby("ticker"):
            ax.scatter(g[c], g["algo_trad"], s=14, alpha=0.7, color=STOCK_COLORS[t], label=t, edgecolor="none")
        r = w[[c, "algo_trad"]].corr().iloc[0, 1]
        ax.set_xlabel(lab)
        ax.set_title(f"algo_trad vs {'turnover' if c == 'turnover' else 'fleeting orders'}:  r = {r:.2f}")
    axes[0].set_ylabel("algo_trad (deviation from stock mean)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9, markerscale=1.5)
    save(fig, "proxy_validity.png")


def lifetimes():
    raw = load_raw()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bins = np.logspace(-6, 4, 81)
    shares = {}
    for t, msg in raw.items():
        life = order_lifetimes(msg)
        x = life.loc[life["cancel_no_fill"], "life_del"].clip(lower=1e-6)
        h, e = np.histogram(x, bins=bins)
        ax.plot(np.sqrt(e[1:] * e[:-1]), h / len(life) * 100, color=STOCK_COLORS[t], label=t, lw=1.8)
        shares[t] = {"submitted": int(len(life)),
                     "cancelled_unfilled": float(life["cancel_no_fill"].mean()),
                     "cancelled_within_100ms": float((life["cancel_no_fill"] & (life["life_del"] <= 0.1)).mean()),
                     "cancelled_within_1s": float((life["cancel_no_fill"] & (life["life_del"] <= 1)).mean()),
                     "cancelled_within_10ms": float((life["cancel_no_fill"] & (life["life_del"] <= 0.01)).mean())}
    ax.set_xscale("log")
    ax.axvspan(0.15, 0.3, color=LIME, alpha=0.35)
    ax.text(0.21, ax.get_ylim()[1] * 0.92, "human\nreaction\ntime", ha="center", va="top", fontsize=9, color=INK)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                  ["10 µs", "0.1 ms", "1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s", "1000 s"])
    ax.set_xlabel("Time from submission to cancellation (orders cancelled without a fill)")
    ax.set_ylabel("% of submitted orders")
    ax.set_xlim(3e-6, 5e3)
    ax.set_title("Most cancellations happen faster than a human can react")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=10)
    save(fig, "order_lifetimes.png")
    return shares


def width_robustness():
    r = pd.read_csv(RES / "robustness_width.csv")
    r = r[r.group == "High-price"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    deps = [("espread", NAVY), ("qspread", "#3D9BC7"), ("rspread_5m", "#9B7FD4"), ("adv_sel_5m", "#A9BC00")]
    for ax, at, title in [(axes[0], "algo_trad", "Paper’s proxy (algo_trad)"),
                          (axes[1], "log_messages", "Log message traffic")]:
        ax.axhspan(-1.96, 1.96, color="#F3F4F8", zorder=0)
        for d, col in deps:
            s = r[(r["at"] == at) & (r["dep"] == d)].sort_values("width_s")
            ax.plot(s.width_s / 60, s.t, marker="o", ms=7, color=col, label=LABELS[d])
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 15], ["1", "5", "10", "15"])
        ax.set_xlabel("Interval length, minutes")
        ax.set_title(title)
    axes[0].set_ylabel("t-statistic, high-price stocks")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "width_robustness.png")


def main():
    p = pd.read_csv(PROC / "panel_main_prepared.csv")
    p5 = pd.read_csv(PROC / "panel_300s.csv")
    t1 = pd.read_csv(RES / "table1_summary.csv", index_col=0)
    intraday(p)
    decomposition(t1)
    coef_comparison()
    first_stage()
    proxy_validity(p)
    width_robustness()
    shares = lifetimes()

    res = json.load(open(RES / "results.json"))
    info = json.load(open(RES / "data_info.json"))
    summary = {"data": info, "order_lifetimes": shares, "sign_match": res["sign_match"],
               "n_obs_main": res["n_obs"], "main_width_s": res["main_width_s"]}
    json.dump(summary, open(RES / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
