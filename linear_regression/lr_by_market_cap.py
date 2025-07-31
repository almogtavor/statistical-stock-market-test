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
import argparse


EXTREME_COMPANIES_PERCENTS = 10 # setting to 10% means we'll measure the top 10% companies, and lowest 10%
P_LOWEST_CAP_CLIPPING_HIGH = 95
P_LOWEST_CAP_CLIPPING_LOW = 5

CSV = "../fcf_dataset.csv"
HORIZONS = {
    "6M": ("6M_FCFps_growth", "6M_Price_growth"),
    "1Y": ("1Y_FCFps_growth", "1Y_Price_growth"),
    "2Y": ("2Y_FCFps_growth", "2Y_Price_growth"),
    "3Y": ("3Y_FCFps_growth", "3Y_Price_growth"),
}
# Load data
df_full = pd.read_csv(CSV, parse_dates=["Report Date"])

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--single-panel", action="store_true",
                    help="Overlay all horizons' All Samples results on one panel")
args = parser.parse_args()

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


# store all-samples results for possible overlay
all_samples_results = {}
all_horizon_results = {}

# run separately for each horizon
for horizon_label, (fcfps_col, price_col) in HORIZONS.items():
    # we need the growth columns plus Market_Cap
    df = df_full[["Ticker", "Market_Cap", fcfps_col, price_col]].dropna()

    # convert to % points
    fcf_pct_col = f"{horizon_label}_FCF_pct"
    price_pct_col = f"{horizon_label}_Price_pct"
    df[fcf_pct_col] = df[fcfps_col] * 100
    df[price_pct_col] = df[price_col] * 100

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

    # run for all tiers, but override percentiles for micro-caps
    results = {}
    for name, idx in tiers.items():
        sub = df.loc[idx]
        x_vals = sub[fcf_pct_col].values
        y_vals = sub[price_pct_col].values
        if name == "Micro-caps":
            r = clip_and_ols(sub[fcf_pct_col].values,
                             sub[price_pct_col].values,
                             p_low=P_LOWEST_CAP_CLIPPING_LOW, p_high=P_LOWEST_CAP_CLIPPING_HIGH)
        else:
            r = clip_and_ols(sub[fcf_pct_col].values,
                             sub[price_pct_col].values)
        if r:
            total = len(x_vals)
            n_after_clipping = r["n"]
            clipped = total - n_after_clipping
            clipped_pct = clipped / total * 100
            results[name] = r
            # Each sample is horizon FCF change of a stock and the change in its price same horizon after
            # same stock can appear multiple times
            print(f"\n=== {horizon_label} {name} === Total samples: {total}, Retained after clipping: {n_after_clipping}, "
                  f"Clipped: {clipped} ({clipped_pct:.1f}%),"
                  f" β₁={r['b1']:.6f}, p_value={r['p_value']:.3g}")

    # stash all-samples if needed later for overlay
    if "All Samples" in results:
        all_samples_results[horizon_label] = results["All Samples"]

    # Plot per-horizon (same as before)
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

    ax.set_title(f"{horizon_label} Price Change vs FCF Growth by Market Cap Tier")
    ax.set_xlabel(f"{horizon_label} FCFps Growth (%)")
    ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    all_horizon_results[horizon_label] = results

# optional overlay of all horizons for "All Samples"
# combined panel: 4 horizons in 2x2, still showing cap tiers
if args.single_panel and all_horizon_results:
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    colors = {"All Samples": "black", "Mega-caps": "blue", "Mid-caps": "green", "Micro-caps": "orange"}
    horizon_order = ["6M", "1Y", "2Y", "3Y"]
    for i, horizon_label in enumerate(horizon_order):
        row, col = divmod(i, 2)
        ax = axs[row, col]
        results = all_horizon_results.get(horizon_label, {})
        for name, r in results.items():
            if name != "All Samples" and name != "Mid-caps":
                ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, color=colors[name], label=f"{name} data")
            elif name == "Mid-caps":
                ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, color="black", label=f"{name} data")
            fit_x = np.linspace(*r["xlim"], 200)
            fit_y = r["b0"] + r["b1"] * fit_x
            ax.plot(fit_x, fit_y, color=colors.get(name, "black"), lw=2, label=f"{name} fit")
        ax.set_title(f"{horizon_label} Price Change vs FCF Growth by Market Cap Tier")
        ax.set_xlabel(f"{horizon_label} FCFps Growth (%)")
        ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="small")
    plt.show()
