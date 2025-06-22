#!/usr/bin/env python3
"""
run_ols_by_cap.py

Split into cap tiers (top10%, mid80%, bottom10%, all) and run
2Y Price Δ vs 2Y FCFps Δ OLS (clipped 1–99pct) for each segment.
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

CSV = "../fcf_dataset.csv"

# -----------------------------------------------------------------------------
# 1 — Load & prep
# -----------------------------------------------------------------------------
df = pd.read_csv(CSV, parse_dates=["Report Date"])
# we need the 2-year growth columns plus Market_Cap
df = df[["Ticker","Market_Cap","YoY2_FCFps_growth","YoY2_Price_growth"]].dropna()

# convert to % points
df["FCF2y_pct"]   = df["YoY2_FCFps_growth"]*100
df["Price2y_pct"] = df["YoY2_Price_growth"]*100

# -----------------------------------------------------------------------------
# 2 — define cap‐tier masks
# -----------------------------------------------------------------------------
cap_q10 = df["Market_Cap"].quantile(0.10)
cap_q90 = df["Market_Cap"].quantile(0.90)

tiers = {
    "All Stocks":   df.index,
    "Mega-caps":    df[df["Market_Cap"] >= cap_q90].index,
    "Micro-caps":   df[df["Market_Cap"] <= cap_q10].index,
    "Mid-caps":     df[(df["Market_Cap"]>cap_q10)&(df["Market_Cap"]<cap_q90)].index,
}

# -----------------------------------------------------------------------------
# 3 — helper to clip & regress & report
# -----------------------------------------------------------------------------
def clip_and_ols(x, y, p_low=1, p_high=99):
    xl, xh = np.percentile(x, [p_low, p_high])
    yl, yh = np.percentile(y, [p_low, p_high])
    mask = (x>=xl)&(x<=xh)&(y>=yl)&(y<=yh)
    xc, yc = x[mask], y[mask]
    n = len(xc)
    if n<5: return None
    xb, yb = xc.mean(), yc.mean()
    Sxx = np.sum((xc-xb)**2)
    Sxy = np.sum((xc-xb)*(yc-yb))
    b1 = Sxy/Sxx
    b0 = yb - b1*xb
    resid = yc - (b0 + b1*xc)
    s2 = np.sum(resid**2)/(n-2)
    SE = np.sqrt(s2/Sxx)
    t  = b1/SE
    p  = 2*stats.t.sf(abs(t), df=n-2)
    tcrit = stats.t.ppf(0.975, df=n-2)
    ci = (b1 - tcrit*SE, b1 + tcrit*SE)
    return {"n":n, "b0":b0, "b1":b1, "SE":SE, "t":t, "p":p, "CI":ci,
            "xlim":(xl,xh),"ylim":(yl,yh),"xc":xc,"yc":yc}

# run for all tiers, but override percentiles for micro-caps
results = {}
for name, idx in tiers.items():
    sub = df.loc[idx]
    if name=="Micro-caps":
        r = clip_and_ols(sub["FCF2y_pct"].values,
                         sub["Price2y_pct"].values,
                         p_low=5, p_high=95)
    else:
        r = clip_and_ols(sub["FCF2y_pct"].values,
                         sub["Price2y_pct"].values)
    if r:
        results[name] = r
        print(f"\n=== {name} === n={r['n']}, β₁={r['b1']:.6f}, p={r['p']:.3g}")

# Plot
fig, ax = plt.subplots(figsize=(12,8))
colors = {"All Stocks":"black","Mega-caps":"blue","Mid-caps":"green","Micro-caps":"orange"}

# main cloud + fits
for name, r in results.items():
    if name=="All Stocks":
        ax.scatter(r["xc"], r["yc"], s=10, alpha=0.2, color=colors[name], label=name)
    xs = np.linspace(*r["xlim"], 200)
    ys = r["b0"] + r["b1"]*xs
    ax.plot(xs, ys, color=colors[name], lw=2, label=f"{name} fit")

# inset for microcaps
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# axins = inset_axes(ax, width="30%", height="30%", loc="upper left")
# r = results["Micro-caps"]
# axins.scatter(r["xc"], r["yc"], s=8, alpha=0.3, color=colors["Micro-caps"])
# xs = np.linspace(*r["xlim"], 100)
# axins.plot(xs, r["b0"]+r["b1"]*xs, color=colors["Micro-caps"], lw=2)
# axins.set_title("Micro-caps (5–95pct clip)")
# axins.set_xlim(*r["xlim"])
# axins.set_ylim(*r["ylim"])
# axins.axhline(0, color="grey", lw=1)
# axins.axvline(0, color="grey", lw=1)

ax.set_title("2Y Price Δ vs 2Y FCFps Δ by Cap Tier")
ax.set_xlabel("2-Year FCFps Growth (%)")
ax.set_ylabel("2-Year Forward Price Change (%)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
