"""Builds and runs AT_Liquidity_Walkthrough.ipynb (data walkthrough and results)."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Does algorithmic trading improve liquidity?"),
    md("## Setup"),
    code("try:\n    from google.colab import drive\n    drive.mount('/content/drive')\nexcept ImportError:\n    pass"),
    code("import sys, json\nfrom pathlib import Path\nimport pandas as pd, numpy as np\ncolab_project = Path('/content/drive/MyDrive/Research_Seminar_AT')\nPROJECT = colab_project if colab_project.exists() else Path.cwd()\nsys.path.insert(0, str(PROJECT / 'code'))\npd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)\nRAW, PROC, RES = PROJECT / 'raw_data', PROJECT / 'processed_data', PROJECT / 'results'"),
    md("## Data"),
    code("for f in sorted(RAW.glob('*.csv.gz')):\n    print(f.name, '|', round(f.stat().st_size / 1024**2, 2), 'MB')"),
    code("from at_liquidity.io import load_raw\nraw = load_raw(RAW)\nmsg = raw['AAPL']\nprint('AAPL messages:', len(msg))\nmsg.head()"),
    code("pd.DataFrame({t: m['type'].value_counts().sort_index() for t, m in raw.items()})"),
    md("## Data checks"),
    code("for t, m in raw.items():\n    assert m['time'].is_monotonic_increasing\n    ok = (m['ask1'] > m['bid1']).mean()\n    print(f'{t}: {len(m):>7,} messages, time monotonic, ask > bid in {ok:.2%} of rows')"),
    code("print(json.dumps(json.loads((RES / 'checks.json').read_text()), indent=1))"),
    md("## Trades and spreads"),
    code("from at_liquidity.trades import build_trades, trade_measures\ntr = build_trades(raw['AAPL'])\ntm = trade_measures(tr)\npd.concat([tr[['time','q','size','price','bid','ask','mid','mid_5m']], tm], axis=1).head()"),
    code("ok = tm[['rspread_5m','adv_sel_5m']].notna().all(axis=1)\nprint('max |espread - rspread - adv_sel| =', (tm.espread - tm.rspread_5m - tm.adv_sel_5m)[ok].abs().max())"),
    md("## Panel"),
    code("p = pd.read_csv(PROC / 'panel_main_prepared.csv')\nprint(p.shape)\np[['ticker','clock','messages','algo_trad','qspread','espread','rspread_5m','adv_sel_5m','impulse_response','fleet_100ms']].head()"),
    md("## Summary statistics"),
    code("pd.read_csv(RES / 'table1_summary.csv', index_col=0).round(3)"),
    md("## OLS regressions"),
    code("ols = pd.read_csv(RES / 'ols_algo_trad.csv')\nols[ols.group == 'High-price'][['table','dep','coef','t','paper_coef','paper_t','same_sign']].round(3)"),
    code("msg_ols = pd.read_csv(RES / 'ols_log_messages.csv')\nmsg_ols[msg_ols.group == 'High-price'][['table','dep','coef','t','paper_coef','same_sign']].round(3)"),
    code("pd.DataFrame(json.loads((RES / 'results.json').read_text())['sign_match']).T"),
    md("## Instrumental variables"),
    code("pd.read_csv(RES / 'table2_first_stage.csv').round(3)"),
    code("iv = pd.read_csv(RES / 'iv_2sls.csv')\niv[iv.group == 'High-price'][['at','dep','coef','t','F','paper_coef']].round(3)"),
    md("## Extension: algorithms in the data"),
    code("pd.read_csv(RES / 'at_proxy_correlations_main.csv', index_col=0).round(2)"),
    code("pd.DataFrame(json.loads((RES / 'summary.json').read_text())['order_lifetimes']).T.round(3)"),
    md("## Figures"),
    code("from IPython.display import Image, display\nfor name in ['intraday_activity','intraday_spreads','spread_decomposition','coef_comparison',\n             'first_stage','order_lifetimes','proxy_validity','width_robustness']:\n    display(Image(filename=str(PROJECT / 'figures' / f'{name}.png'), width=900))"),
]


nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "AT_Liquidity_Walkthrough.ipynb"
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, out)
print("written", out)
