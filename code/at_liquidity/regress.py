"""Panel OLS and 2SLS with fixed effects and Driscoll-Kraay standard errors.

The paper uses standard errors robust to cross-sectional and time-series
correlation (Arellano-Bond style). With only five stocks we cannot cluster
by stock, so we use Driscoll & Kraay (1998): residual moments are summed
across stocks in each interval, then Newey-West weighted over time.
"""
import numpy as np
import pandas as pd
from scipy import stats


def winsorize(s, p=0.005):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def _dummies(df, col, prefix):
    return pd.get_dummies(df[col], prefix=prefix, drop_first=True, dtype=float)


def design(df, y, x, controls, fe_stock=True, fe_time="bin"):
    """y, regressor matrix with fixed effects as dummies, time index."""
    cols = [y, x] + list(controls)
    d = df.dropna(subset=cols).copy()
    parts = [d[[x] + list(controls)].astype(float)]
    if fe_stock:
        parts.append(_dummies(d, "ticker", "s"))
    if fe_time:
        parts.append(_dummies(d, fe_time, "t"))
    X = pd.concat(parts, axis=1)
    X.insert(0, "const", 1.0)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    return d, d[y].astype(float).to_numpy(), X


def _dk_cov(Xh, e, time, bread, lags=None):
    T = len(np.unique(time))
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    h = pd.DataFrame(Xh * e[:, None]).groupby(time).sum().sort_index().to_numpy()
    S = h.T @ h
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = h[l:].T @ h[:-l]
        S += w * (G + G.T)
    return bread @ S @ bread


def fit(df, y, x, controls=(), instrument=None, fe_time="bin", fe_stock=True):
    """OLS (instrument=None) or 2SLS of y on x. Returns coefficient on x and stats.

    In 2SLS the instrument replaces x in the instrument set; controls and
    fixed effects instrument themselves, as in the paper.
    """
    cols_needed = [instrument] if instrument else []
    d = df.dropna(subset=cols_needed) if instrument else df
    d, Y, X = design(d, y, x, controls, fe_stock, fe_time)
    Xv = X.to_numpy()
    time = d["bin"].to_numpy()
    if instrument is None:
        Xh = Xv
    else:
        Z = X.copy()
        Z[x] = d[instrument].astype(float).to_numpy()
        Zv = Z.to_numpy()
        Xh = Zv @ np.linalg.lstsq(Zv, Xv, rcond=None)[0]       # fitted X
    bread = np.linalg.pinv(Xh.T @ Xh)
    beta = bread @ Xh.T @ Y
    e = Y - Xv @ beta                                           # structural residuals
    V = _dk_cov(Xh, e, time, bread)
    j = list(X.columns).index(x)
    b, se = beta[j], np.sqrt(V[j, j])
    T = len(np.unique(time))
    out = {"coef": b, "se": se, "t": b / se, "p": 2 * stats.t.sf(abs(b / se), T - 1),
           "n": len(Y), "n_time": T}
    out["controls"] = {c: beta[list(X.columns).index(c)] for c in controls if c in X.columns}
    return out


def first_stage(df, x, instrument, controls=(), fe_time="halfhour"):
    """Regression of the AT measure on the instrument (Table II analogue)."""
    r = fit(df, x, instrument, controls, None, fe_time)
    r["F"] = r["t"] ** 2
    return r


def stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""
