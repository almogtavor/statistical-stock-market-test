#!/usr/bin/env python3
"""
lr_by_market_cap_robust.py

Enhanced version that splits into cap tiers (top10%, mid80%, bottom10%, all) and runs
Price Δ vs FCFps Δ analysis for each segment, including:
- OLS regression with R^2, RSS
- Robust regression (Huber regression)
- Comprehensive statistical metrics
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


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
parser.add_argument("--show-robust", action="store_true", default=True,
                    help="Show robust regression lines alongside OLS")
parser.add_argument("--save-plots", action="store_true",
                    help="Save plots to files instead of showing them")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Enhanced helper to clip & regress & report with R^2 and RSS
# -----------------------------------------------------------------------------
def enhanced_regression_analysis(x, y, p_low=0.9, p_high=99.1):
    """
    Perform both OLS and robust regression with comprehensive metrics
    """
    # Clip outliers
    x_clip_min, x_clip_max = np.percentile(x, [p_low, p_high])
    y_clip_min, y_clip_max = np.percentile(y, [p_low, p_high])
    mask = (x >= x_clip_min) & (x <= x_clip_max) & (y >= y_clip_min) & (y <= y_clip_max)
    x_clipped, y_clipped = x[mask], y[mask]
    n = len(x_clipped)
    
    if n < 5: 
        return None
    
    # Reshape for sklearn
    X = x_clipped.reshape(-1, 1)
    
    # OLS Regression (manual calculation for consistency with original)
    x_mean, y_mean = x_clipped.mean(), y_clipped.mean()
    Sxx = np.sum((x_clipped - x_mean) ** 2)
    Sxy = np.sum((x_clipped - x_mean) * (y_clipped - y_mean))
    
    if Sxx == 0:
        return None
        
    ols_slope = Sxy / Sxx
    ols_intercept = y_mean - ols_slope * x_mean
    
    # Predictions and residuals
    y_pred_ols = ols_intercept + ols_slope * x_clipped
    residuals_ols = y_clipped - y_pred_ols
    
    # RSS (Residual Sum of Squares)
    rss_ols = np.sum(residuals_ols ** 2)
    
    # R^2 calculation
    tss = np.sum((y_clipped - y_mean) ** 2)  # Total Sum of Squares
    r2_ols = 1 - (rss_ols / tss) if tss > 0 else 0
    
    # Standard error and statistical tests
    residual_var = rss_ols / (n - 2)
    SE = np.sqrt(residual_var / Sxx)
    t_stat = ols_slope / SE if SE > 0 else 0
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 2) if SE > 0 else 1
    tcrit = stats.t.ppf(0.975, df=n - 2)
    ci = (ols_slope - tcrit * SE, ols_slope + tcrit * SE) if SE > 0 else (ols_slope, ols_slope)
    
    # Robust Regression (Huber)
    try:
        huber = HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.0001)
        huber.fit(X, y_clipped)
        robust_slope = huber.coef_[0]
        robust_intercept = huber.intercept_
        
        # Robust predictions and metrics
        y_pred_robust = robust_intercept + robust_slope * x_clipped
        residuals_robust = y_clipped - y_pred_robust
        rss_robust = np.sum(residuals_robust ** 2)
        r2_robust = r2_score(y_clipped, y_pred_robust)
        
    except Exception as e:
        print(f"Robust regression failed: {e}")
        robust_slope = ols_slope
        robust_intercept = ols_intercept
        rss_robust = rss_ols
        r2_robust = r2_ols
    
    return {
        "n": n,
        # OLS results
        "ols_intercept": ols_intercept,
        "ols_slope": ols_slope,
        "ols_SE": SE,
        "ols_t": t_stat,
        "ols_p_value": p_value,
        "ols_CI": ci,
        "ols_r2": r2_ols,
        "ols_rss": rss_ols,
        # Robust results
        "robust_intercept": robust_intercept,
        "robust_slope": robust_slope,
        "robust_r2": r2_robust,
        "robust_rss": rss_robust,
        # Data bounds and clipped data
        "xlim": (x_clip_min, x_clip_max),
        "ylim": (y_clip_min, y_clip_max),
        "x_clipped": x_clipped,
        "y_clipped": y_clipped
    }


def format_results_table(results_dict, horizon_label):
    """
    Print a nicely formatted table of results
    """
    print(f"\n{'='*80}")
    print(f"REGRESSION RESULTS FOR {horizon_label}")
    print(f"{'='*80}")
    
    headers = ["Tier", "N", "OLS β₁", "OLS R^2", "OLS RSS", "Robust β₁", "Robust R^2", "Robust RSS", "p-value"]
    print(f"{headers[0]:<12} {headers[1]:<6} {headers[2]:<10} {headers[3]:<8} {headers[4]:<12} {headers[5]:<10} {headers[6]:<8} {headers[7]:<12} {headers[8]:<10}")
    print("-" * 80)
    
    for name, r in results_dict.items():
        if r is None:
            continue
        print(f"{name:<12} {r['n']:<6} {r['ols_slope']:<10.4f} {r['ols_r2']:<8.3f} {r['ols_rss']:<12.1f} "
              f"{r['robust_slope']:<10.4f} {r['robust_r2']:<8.3f} {r['robust_rss']:<12.1f} {r['ols_p_value']:<10.3g}")


# Store results for analysis
all_samples_results = {}
all_horizon_results = {}

# Run separately for each horizon
for horizon_label, (fcfps_col, price_col) in HORIZONS.items():
    print(f"\n\nProcessing {horizon_label} horizon...")
    
    # We need the growth columns plus Market_Cap
    df = df_full[["Ticker", "Market_Cap", fcfps_col, price_col]].dropna()

    # Convert to % points
    fcf_pct_col = f"{horizon_label}_FCF_pct"
    price_pct_col = f"{horizon_label}_Price_pct"
    df[fcf_pct_col] = df[fcfps_col] * 100
    df[price_pct_col] = df[price_col] * 100

    # -----------------------------------------------------------------------------
    # Define cap tiers
    # -----------------------------------------------------------------------------
    largest_market_cap_stocks = df["Market_Cap"].quantile(EXTREME_COMPANIES_PERCENTS / 100)
    smallest_market_cap_stocks = df["Market_Cap"].quantile(1 - (EXTREME_COMPANIES_PERCENTS / 100))

    tiers = {
        "All Samples": df.index,
        "Mega-caps": df[df["Market_Cap"] >= smallest_market_cap_stocks].index,
        "Micro-caps": df[df["Market_Cap"] <= largest_market_cap_stocks].index,
        "Mid-caps": df[(df["Market_Cap"] > largest_market_cap_stocks) & (df["Market_Cap"] < smallest_market_cap_stocks)].index,
    }

    # Run enhanced analysis for all tiers
    results = {}
    for name, idx in tiers.items():
        sub = df.loc[idx]
        x_vals = sub[fcf_pct_col].values
        y_vals = sub[price_pct_col].values
        
        if name == "Micro-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                           p_low=P_LOWEST_CAP_CLIPPING_LOW, 
                                           p_high=P_LOWEST_CAP_CLIPPING_HIGH)
        elif name == "Mega-caps" or name == "Mid-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                            p_low=1, 
                                            p_high=99)
        else:
            r = enhanced_regression_analysis(x_vals, y_vals)

        if r:
            results[name] = r

    # Print formatted results table
    format_results_table(results, horizon_label)

    # Store results for plotting
    if "All Samples" in results:
        all_samples_results[horizon_label] = results["All Samples"]
    all_horizon_results[horizon_label] = results

    # Create enhanced plot with both OLS and robust lines
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = {"All Samples": "black", "Mega-caps": "blue", "Mid-caps": "green", "Micro-caps": "orange"}
    
    # Plot data points and regression lines
    for name, r in results.items():
        if r is None:
            continue
            
        # Scatter plot
        if name != "Mid-caps":
            ax.scatter(r["x_clipped"], r["y_clipped"], s=15, alpha=0.4, 
                      color=colors[name], label=f"{name} data (n={r['n']})")
        else:
            ax.scatter(r["x_clipped"], r["y_clipped"], s=15, alpha=0.4, 
                      color="black", label=f"{name} data (n={r['n']})")
        
        # OLS fit line
        fit_x = np.linspace(*r["xlim"], 200)
        ols_fit_y = r["ols_intercept"] + r["ols_slope"] * fit_x
        ax.plot(fit_x, ols_fit_y, color=colors[name], lw=2, 
                label=f"{name} OLS (R^2={r['ols_r2']:.3f})")
        
        # Robust fit line (if enabled)
        if args.show_robust:
            robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
            ax.plot(fit_x, robust_fit_y, color=colors[name], lw=2, linestyle='--',
                    label=f"{name} Robust (R^2={r['robust_r2']:.3f})")

    ax.set_title(f"{horizon_label} Price Change vs FCF Growth by Market Cap Tier\n"
                f"(OLS {'and Robust ' if args.show_robust else ''}Regression with R^2 and RSS)")
    ax.set_xlabel(f"{horizon_label} FCFps Growth (%)")
    ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if args.save_plots:
        plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression.png")
    else:
        plt.show()

# Optional: Combined panel showing all horizons
if args.single_panel and all_horizon_results:
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    colors = {"All Samples": "black", "Mega-caps": "blue", "Mid-caps": "green", "Micro-caps": "orange"}
    horizon_order = ["6M", "1Y", "2Y", "3Y"]
    
    for i, horizon_label in enumerate(horizon_order):
        row, col = divmod(i, 2)
        ax = axs[row, col]
        results = all_horizon_results.get(horizon_label, {})
        
        for name, r in results.items():
            if r is None:
                continue
                
            # Scatter plot
            if name != "Mid-caps":
                ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, 
                          color=colors[name], label=f"{name} data")
            else:
                ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, 
                          color="black", label=f"{name} data")
            
            # Regression lines
            fit_x = np.linspace(*r["xlim"], 200)
            ols_fit_y = r["ols_intercept"] + r["ols_slope"] * fit_x
            ax.plot(fit_x, ols_fit_y, color=colors.get(name, "black"), lw=2, 
                    label=f"{name} OLS (R^2={r['ols_r2']:.3f})")
            
            if args.show_robust:
                robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
                ax.plot(fit_x, robust_fit_y, color=colors.get(name, "black"), 
                       lw=2, linestyle='--', label=f"{name} Robust")
        
        ax.set_title(f"{horizon_label} Price Change vs FCF Growth by Market Cap Tier")
        ax.set_xlabel(f"{horizon_label} FCFps Growth (%)")
        ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small")
    
    if args.save_plots:
        plt.savefig("all_horizons_market_cap_robust_regression.png", 
                   dpi=300, bbox_inches='tight')
        print("Saved combined plot: all_horizons_market_cap_robust_regression.png")
    else:
        plt.show()

# Summary statistics across all horizons
print(f"\n{'='*80}")
print("SUMMARY: R^2 COMPARISON ACROSS HORIZONS")
print(f"{'='*80}")
print(f"{'Horizon':<10} {'All Samples OLS R^2':<20} {'All Samples Robust R^2':<22} {'Mega-caps OLS R^2':<18} {'Micro-caps OLS R^2':<18}")
print("-" * 80)

for horizon_label in HORIZONS.keys():
    results = all_horizon_results.get(horizon_label, {})
    all_ols_r2 = results.get("All Samples", {}).get("ols_r2", 0)
    all_robust_r2 = results.get("All Samples", {}).get("robust_r2", 0)
    mega_ols_r2 = results.get("Mega-caps", {}).get("ols_r2", 0)
    micro_ols_r2 = results.get("Micro-caps", {}).get("ols_r2", 0)
    
    print(f"{horizon_label:<10} {all_ols_r2:<20.4f} {all_robust_r2:<22.4f} {mega_ols_r2:<18.4f} {micro_ols_r2:<18.4f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
