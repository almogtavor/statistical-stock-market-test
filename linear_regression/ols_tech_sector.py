#!/usr/bin/env python3
"""
OLS slopes: All vs. Technology (heuristic) firms,
plotted together on winsorized, trimmed data.
"""

import os
import pandas as pd
import numpy as np
import simfin as sf
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# ─── Setup SimFin ──────────────────────────────────────────────────────────
load_dotenv()
sf.set_api_key(os.getenv('SIMFIN_API_KEY', 'free'))
sf.set_data_dir(str(Path('../simfin_data')))

# ─── 1) Load FCF dataset ───────────────────────────────────────────────────
df = pd.read_csv("../fcf_dataset.csv")[[
    "Ticker","YoY_Price_growth","YoY_FCFps_growth","Market_Cap"
]].replace([np.inf,-np.inf], np.nan).dropna()



import statsmodels.api as sm

df['Up'] = (df['YoY_Price_growth'] > 0).astype(int)
X = sm.add_constant(df['YoY_FCFps_growth'])
logit = sm.Logit(df['Up'], X).fit(disp=False)
print(logit.params['YoY_FCFps_growth'], logit.pvalues['YoY_FCFps_growth'])



# ─── 2) Winsorise to 1–99% ────────────────────────────────────────────────
for c in ["YoY_Price_growth","YoY_FCFps_growth"]:
    lo, hi = df[c].quantile([0.01,0.99])
    df[c]  = df[c].clip(lo, hi)

# ─── 3) Trim & rescale ────────────────────────────────────────────────────
df = df[np.abs(df["YoY_FCFps_growth"]) <= 1].copy()
df["YoY_FCFps_growth"] *= 100  # now in percentage points

# ─── 4) Heuristic tech filter via Business Summary ───────────────────────
companies = (
    sf.load_companies(market="US", refresh_days=365)
      .reset_index()[["Ticker","Business Summary"]]
)
# Mark as tech if summary mentions "technology"
companies["IsTech"] = (
    companies["Business Summary"]
      .str.contains("technology", case=False, na=False)
)
df = df.merge(companies[["Ticker","IsTech"]], on="Ticker", how="left")
df = df.dropna(subset=["IsTech"])  # drop if no summary

# ─── 5) OLS helper ────────────────────────────────────────────────────────
def ols_params(x, y):
    x̄,ȳ = x.mean(), y.mean()
    Sxx  = ((x-x̄)**2).sum()
    Sxy  = ((x-x̄)*(y-ȳ)).sum()
    β1   = Sxy/Sxx
    β0   = ȳ - β1*x̄
    return β0, β1

def ols_stats(x, y):
    β0, β1 = ols_params(x,y)
    ŷ      = β0 + β1*x
    resid   = y - ŷ
    s2      = (resid**2).sum()/(len(x)-2)
    SE      = np.sqrt(s2/((x-x.mean())**2).sum())
    t_stat  = β1/SE
    p_val   = 2*stats.t.sf(abs(t_stat), df=len(x)-2)
    return β0, β1, SE, t_stat, p_val

# ─── 6) Fit All & Tech ────────────────────────────────────────────────────
b0_all,  b1_all,  SE_all,  t_all,  p_all  = ols_stats(
    df["YoY_FCFps_growth"], df["YoY_Price_growth"]
)
tech_df = df[df["IsTech"]]
b0_tech, b1_tech, SE_tech, t_tech, p_tech = ols_stats(
    tech_df["YoY_FCFps_growth"], tech_df["YoY_Price_growth"]
)




# ─── 7) Plot All vs. Technology ────────────────────────────────────────────
plt.figure(figsize=(10,6))

# scatter all in light gray
plt.scatter(df["YoY_FCFps_growth"], df["YoY_Price_growth"],
            s=10, alpha=0.2, color="gray", label="All firms")

# highlight tech in blue
plt.scatter(tech_df["YoY_FCFps_growth"], tech_df["YoY_Price_growth"],
            s=15, alpha=0.3, color="tab:blue", label="Technology")

# regression lines
x_line = np.linspace(df["YoY_FCFps_growth"].min(),
                     df["YoY_FCFps_growth"].max(), 200)
plt.plot(x_line, b0_all  + b1_all * x_line,  color="black",   lw=2,
         label=f"All: β₁={b1_all:.4f}, p={p_all:.3g}")
plt.plot(x_line, b0_tech + b1_tech* x_line, color="tab:blue", lw=2,
         label=f"Tech: β₁={b1_tech:.4f}, p={p_tech:.3g}")

plt.axhline(0, color="k", lw=1, ls="--")
plt.axvline(0, color="k", lw=1, ls="--")
plt.xlabel("YoY FCF per Share Growth (pp)")
plt.ylabel("YoY Price Growth")
plt.title("FCF–Price Relation: All Firms vs. Technology (heuristic)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
