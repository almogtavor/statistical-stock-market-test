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


EXTREME_COMPANIES_PERCENTS = 10 # setting to 10% means we'll measure the top 10% companies, and lowest 10%
P_LOWEST_CAP_CLIPPING_HIGH = 96
P_LOWEST_CAP_CLIPPING_LOW = 4

CSV = "../fcf_dataset.csv"
HORIZONS = {
    "6M": ("Yo6M_FCFps_growth", "Yo6M_Price_growth"),
    "1Y": ("YoY_FCFps_growth", "YoY_Price_growth"),
    "2Y": ("YoY2_FCFps_growth", "YoY2_Price_growth"),
    "3Y": ("YoY3_FCFps_growth", "YoY3_Price_growth"),
}
# Load data
df = pd.read_csv(CSV, parse_dates=["Report Date"])
# we need the 2-year growth columns plus Market_Cap
df = df[["Ticker", "Market_Cap", "2Y_FCFps_growth", "2Y_Price_growth"]].dropna()

# convert to % points
df["FCF2y_pct"] = df["2Y_FCFps_growth"] * 100
df["Price2y_pct"] = df["2Y_Price_growth"] * 100

# -----------------------------------------------------------------------------
# define cap‐tiers
# -----------------------------------------------------------------------------
largest_market_cap_stocks = df["Market_Cap"].quantile(EXTREME_COMPANIES_PERCENTS / 100)
smallest_market_cap_stocks = df["Market_Cap"].quantile(1 - (EXTREME_COMPANIES_PERCENTS / 100))

tiers = {
    "All Samples": df.index,
    "Mega-caps": df[df["Market_Cap"] >= smallest_market_cap_stocks].index,
    "Micro-caps": df[df["Market_Cap"] <= largest_market_cap_stocks].index,
    "Mid-caps": df[(df["Market_Cap"] > largest_market_cap_stocks) & (df["Market_Cap"] < smallest_market_cap_stocks)].index,
}


# -----------------------------------------------------------------------------
# helper to clip & regress & report
# -----------------------------------------------------------------------------
def clip_and_ols(x, y, p_low=0.9, p_high=99.1):
    x_clip_min, x_clip_max = np.percentile(x, [p_low, p_high])
    y_clip_min, y_clip_max = np.percentile(y, [p_low, p_high])
    mask = (x >= x_clip_min) & (x <= x_clip_max) & (y >= y_clip_min) & (y <= y_clip_max)
    x_clipped, y_clipped = x[mask], y[mask]
    n = len(x_clipped)
    if n < 5: return None
    x_mean, y_mean = x_clipped.mean(), y_clipped.mean()
    Sxx = np.sum((x_clipped - x_mean) ** 2)
    Sxy = np.sum((x_clipped - x_mean) * (y_clipped - y_mean))
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    residuals = y_clipped - (intercept + slope * x_clipped)
    residual_var = np.sum(residuals ** 2) / (n - 2)
    SE = np.sqrt(residual_var / Sxx)
    t_stat = slope / SE
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 2)
    tcrit = stats.t.ppf(0.975, df=n - 2)
    ci = (slope - tcrit * SE, slope + tcrit * SE)
    return {
        "n": n,
        "b0": intercept,
        "b1": slope,
        "SE": SE,
        "t": t_stat,
        "p_value": p_value,
        "CI": ci,
        "xlim": (x_clip_min, x_clip_max),
        "ylim": (y_clip_min, y_clip_max),
        "x_clipped": x_clipped,
        "y_clipped": y_clipped
    }


# run for all tiers, but override percentiles for micro-caps
results = {}
for name, idx in tiers.items():
    sub = df.loc[idx]
    x_vals = sub["FCF2y_pct"].values
    y_vals = sub["Price2y_pct"].values
    if name == "Micro-caps":
        r = clip_and_ols(sub["FCF2y_pct"].values,
                         sub["Price2y_pct"].values,
                         p_low=P_LOWEST_CAP_CLIPPING_LOW, p_high=P_LOWEST_CAP_CLIPPING_HIGH)
    else:
        r = clip_and_ols(sub["FCF2y_pct"].values,
                         sub["Price2y_pct"].values)
    if r:
        total = len(x_vals)
        n_after_clipping = r["n"]
        clipped = total - n_after_clipping
        clipped_pct = clipped / total * 100
        results[name] = r
        # Each sample is 2 years of fcf change of a stock and the change in its price 2 years after
        # same stock can appear multiple times
        print(f"\n=== {name} === Total samples: {total}, Retained after clipping: {n_after_clipping}, "
              f"Clipped: {clipped} ({clipped_pct:.1f}%),"
              f" β₁={r['b1']:.6f}, p_value={r['p_value']:.3g}")

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = {"All Samples": "black", "Mega-caps": "blue", "Mid-caps": "green", "Micro-caps": "orange"}

# main cloud + fits
for name, r in results.items():
    if name != "All Samples" and name != "Mid-caps":
        ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, color=colors[name], label=f"{name} data")
    elif name == "Mid-caps":
        ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, color="black", label=f"{name} data")
    fit_x = np.linspace(*r["xlim"], 200)
    fit_y = r["b0"] + r["b1"] * fit_x
    ax.plot(fit_x, fit_y, color=colors[name], lw=2, label=f"{name} fit")

ax.set_title("2Y Price Change vs FCF Growth by Market Cap Tier")
ax.set_xlabel("2-Year FCFps Growth (%)")
ax.set_ylabel("2-Year Forward Price Change (%)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
