"""Hasbrouck (1991a,b) VAR: how much of price discovery comes through trades.

Equations (8)-(11) of the paper, estimated in trade time:
    r_t = sum_{i=1..10} a_i r_{t-i} + sum_{i=0..10} b_i q_{t-i} + e_r
    q_t = sum_{i=1..10} c_i r_{t-i} + sum_{i=1..10} d_i q_{t-i} + e_q
r_t is the log change of the midpoint from just before trade t to just
before trade t+1, so the contemporaneous q_t can move r_t.
"""
import numpy as np

LAGS = 10
HORIZON = 50   # trades after the shock; the responses are flat long before


def _lagmat(x, lags, start):
    n = len(x)
    return np.column_stack([x[lags - i:n - i] for i in range(start, lags + 1)])


def fit_var(r, q, lags=LAGS):
    """OLS of both equations. Returns coefficients and residual variances."""
    n = len(r)
    y_r, y_q = r[lags:], q[lags:]
    X_r = np.column_stack([np.ones(n - lags), _lagmat(r, lags, 1), _lagmat(q, lags, 0)])
    X_q = np.column_stack([np.ones(n - lags), _lagmat(r, lags, 1), _lagmat(q, lags, 1)])
    br, *_ = np.linalg.lstsq(X_r, y_r, rcond=None)
    bq, *_ = np.linalg.lstsq(X_q, y_q, rcond=None)
    s2_r = np.var(y_r - X_r @ br, ddof=X_r.shape[1])
    s2_q = np.var(y_q - X_q @ bq, ddof=X_q.shape[1])
    a = br[1:1 + lags]                 # r lags in r equation
    b = br[1 + lags:]                  # q_t, q_{t-1}, ... in r equation (11 terms)
    c = bq[1:1 + lags]                 # r lags in q equation
    d = bq[1 + lags:]                  # q lags in q equation
    return a, b, c, d, s2_r, s2_q


def cumulative_response(a, b, c, d, shock, horizon=HORIZON):
    """Cumulative midpoint response to a unit shock in e_r ('r') or e_q ('q')."""
    lags = len(a)
    r = np.zeros(horizon + lags)
    q = np.zeros(horizon + lags)
    for t in range(lags, horizon + lags):
        past_r = r[t - lags:t][::-1]          # r_{t-1}, ..., r_{t-10}
        past_q = q[t - lags:t][::-1]
        q[t] = c @ past_r + d @ past_q + (1.0 if (shock == "q" and t == lags) else 0.0)
        r[t] = a @ past_r + b[0] * q[t] + b[1:] @ past_q + (1.0 if (shock == "r" and t == lags) else 0.0)
    return r[lags:].sum()


def decompose(r, q, lags=LAGS):
    """Impulse response (bps) and the two parts of efficient-price variance.

    Returned standard deviations are per interval: per-trade variances are
    multiplied by the number of trades, as the paper does per day.
    """
    r = np.asarray(r, float)
    q = np.asarray(q, float)
    a, b, c, d, s2_r, s2_q = fit_var(r, q, lags)
    sum_a = cumulative_response(a, b, c, d, "r")
    sum_b = cumulative_response(a, b, c, d, "q")
    n = len(r)
    var_trade = sum_b ** 2 * s2_q * n
    var_nontrade = sum_a ** 2 * s2_r * n
    return {
        "impulse_response": sum_b * 1e4,                       # bps per unit trade shock
        "sd_tradecorr": np.sqrt(var_trade) * 1e4,               # bps per interval
        "sd_nontradecorr": np.sqrt(var_nontrade) * 1e4,
        "share_tradecorr": var_trade / (var_trade + var_nontrade),
        "n_trades_var": n,
    }
