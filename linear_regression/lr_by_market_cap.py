#!/usr/bin/env python3
"""
OLS slopes for All / Small / Mid / Large market-cap buckets,
plotted together.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CSV = "fcf_dataset.csv"
df  = pd.read_csv(CSV)

# ─── 1) Load & basic clean ───────────────────────────────────────────────
cols = ["YoY_Price_growth","YoY_FCFps_growth","Market_Cap"]
df   = df[cols].replace([np.inf,-np.inf],np.nan).dropna()

# ─── 2) Winsorise to 1–99% ──────────────────────────────────────────────
for c in ["YoY_Price_growth","YoY_FCFps_growth"]:
    lo,hi = df[c].quantile([0.01,0.99])
    df[c] = df[c].clip(lo,hi)

# ─── 3) Trim to |FCF growth| ≤100% and rescale to pct pts ─────────────
mask = np.abs(df["YoY_FCFps_growth"]) <= 1
df   = df.loc[mask].copy()
df["YoY_FCFps_growth"] *= 100

# ─── 4) Define market-cap buckets ──────────────────────────────────────
q10, q90 = df["Market_Cap"].quantile([0.1,0.9])
df["CapBucket"] = np.where(df["Market_Cap"] < q10, "Small Cap",
                   np.where(df["Market_Cap"] > q90, "Large Cap", "Mid Cap"))

# ─── 5) OLS helper ──────────────────────────────────────────────────────
def ols_params(x, y):
    x̄,ȳ = x.mean(), y.mean()
    Sxx  = ((x-x̄)**2).sum()
    Sxy  = ((x-x̄)*(y-ȳ)).sum()
    β1   = Sxy/Sxx
    β0   = ȳ - β1*x̄
    return β0, β1

# ─── 6) Fit All + by bucket ─────────────────────────────────────────────
fits = {}
for name, sub in [("All", df)] + list(df.groupby("CapBucket")):
    b0,b1 = ols_params(sub["YoY_FCFps_growth"], sub["YoY_Price_growth"])
    fits[name] = (b0,b1)

# ─── 7) Plot ────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
x = df["YoY_FCFps_growth"]
y = df["YoY_Price_growth"]
plt.scatter(x, y, s=10, alpha=0.2, color="gray", label="Data")

# colors for the lines
colors = {"All":"black","Small Cap":"tab:blue","Mid Cap":"tab:green","Large Cap":"tab:red"}

x_line = np.linspace(x.min(), x.max(), 200)
for name,(b0,b1) in fits.items():
    y_line = b0 + b1*x_line
    plt.plot(x_line, y_line, color=colors[name], lw=2, label=f"{name} slope")

plt.axhline(0, color='k', lw=1, ls='--')
plt.axvline(0, color='k', lw=1, ls='--')
plt.xlabel("YoY FCF per Share Growth (pp)")
plt.ylabel("YoY Price Growth")
plt.title("OLS slopes by Market-Cap Bucket")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
