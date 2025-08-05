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

# Index ticker lists
NASDAQ100_TICKERS = ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS']

DOW30_TICKERS = ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT']

SP500_TICKERS = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']

CSV = "../fcf_dataset.csv"
HORIZONS = {
    "6M": ("6M_FCFps_growth", "6M_Price_growth"),
    "1Y": ("1Y_FCFps_growth", "1Y_Price_growth"),
    "2Y": ("2Y_FCFps_growth", "2Y_Price_growth"),
    "3Y": ("3Y_FCFps_growth", "3Y_Price_growth"),
}

# FCF Yield horizons (using FCF yield instead of growth)
YIELD_HORIZONS = {
    "6M": ("FCF_yield", "6M_Price_growth"),
    "1Y": ("FCF_yield", "1Y_Price_growth"),
    "2Y": ("FCF_yield", "2Y_Price_growth"),
    "3Y": ("FCF_yield", "3Y_Price_growth"),
}

# Net Income Growth horizons (using Net Income growth rates)
NET_INCOME_HORIZONS = {
    "6M": ("6M_NetIncome_growth", "6M_Price_growth"),
    "1Y": ("1Y_NetIncome_growth", "1Y_Price_growth"),
    "2Y": ("2Y_NetIncome_growth", "2Y_Price_growth"),
    "3Y": ("3Y_NetIncome_growth", "3Y_Price_growth"),
}

# Load data
df_full = pd.read_csv(CSV, parse_dates=["Report Date"])

# Calculate Net Income growth rates for different horizons
# Using pct_change with periods that align with the price growth horizons
# Sort by Ticker and Report Date to ensure proper time series ordering
df_full = df_full.sort_values(['Ticker', 'Report Date'])
df_full["6M_NetIncome_growth"] = df_full.groupby('Ticker')["Net Income"].pct_change(2)  # 2 quarters = 6 months
df_full["1Y_NetIncome_growth"] = df_full.groupby('Ticker')["Net Income"].pct_change(4)   # 4 quarters = 1 year
df_full["2Y_NetIncome_growth"] = df_full.groupby('Ticker')["Net Income"].pct_change(8)   # 8 quarters = 2 years
df_full["3Y_NetIncome_growth"] = df_full.groupby('Ticker')["Net Income"].pct_change(12)  # 12 quarters = 3 years

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
parser.add_argument("--no-plots", action="store_true",
                    help="Skip all plotting and show only statistical results")
parser.add_argument("--nasdaq100-only", action="store_true",
                    help="Restrict analysis to Nasdaq-100 (QQQ) tickers only")
parser.add_argument("--dow30-only", action="store_true",
                    help="Restrict analysis to Dow Jones 30 tickers only")
parser.add_argument("--sp500-only", action="store_true",
                    help="Restrict analysis to S&P 500 tickers only")
parser.add_argument("--use-fcf-yield", action="store_true",
                    help="Use FCF Yield (FCF/EV) instead of FCF growth rates")
parser.add_argument("--use-net-income-growth", action="store_true",
                    help="Use Net Income Growth rates instead of FCF growth rates")
args = parser.parse_args()

# Calculate FCF Yield if using yield mode
if args.use_fcf_yield:
    # Calculate FCF Yield as FCF / Market_Cap (expressed as percentage)
    # Using Market Cap as proxy for EV since EV data is not available
    df_full["FCF_yield"] = (df_full["FCF"] / df_full["Market_Cap"]) * 100

# Calculate Net Income Growth rates if using net income growth mode
if args.use_net_income_growth:
    # Net income growth rates are already calculated above after loading the data
    # They represent the percentage change in net income over the corresponding time periods
    # This should provide better correlation with price changes as both are growth metrics
    pass  # Growth rates already calculated above

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
    
    # Calculate Pearson and Spearman correlation coefficients
    pearson_corr, _ = stats.pearsonr(x_clipped, y_clipped)
    spearman_corr, _ = stats.spearmanr(x_clipped, y_clipped)
    
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
        "y_clipped": y_clipped,
        # Correlation coefficients
        "correlations": {
            "pearson": pearson_corr,
            "spearman": spearman_corr
        }
    }


def format_results_table(results_dict, horizon_label):
    """
    Print a nicely formatted table of results
    """
    print(f"\n{'='*80}")
    print(f"REGRESSION RESULTS FOR {horizon_label}")
    print(f"{'='*80}")
    
    # Update the formatted results table to include correlation coefficients
    headers = ["Tier", "N", "OLS Beta1", "OLS R^2", "OLS RSS", "Robust Beta1", "Robust R^2", "Robust RSS", "p-value", "Pearson Corr", "Spearman Corr"]
    print(f"{headers[0]:<12} {headers[1]:<6} {headers[2]:<10} {headers[3]:<8} {headers[4]:<12} {headers[5]:<10} {headers[6]:<8} {headers[7]:<12} {headers[8]:<10} {headers[9]:<14} {headers[10]:<14}")
    print("-" * 120)

    for name, r in results_dict.items():
        if r is None:
            continue
        print(f"{name:<12} {r['n']:<6} {r['ols_slope']:<10.4f} {r['ols_r2']:<8.3f} {r['ols_rss']:<12.1f} "
              f"{r['robust_slope']:<10.4f} {r['robust_r2']:<8.3f} {r['robust_rss']:<12.1f} {r['ols_p_value']:<10.3g} "
              f"{r['correlations']['pearson']:<14.3f} {r['correlations']['spearman']:<14.3f}")


# Store results for analysis
all_samples_results = {}
all_horizon_results = {}

# Choose horizons based on mode
if args.use_fcf_yield:
    horizons_to_use = YIELD_HORIZONS
    analysis_mode = "FCF Yield"
elif args.use_net_income_growth:
    horizons_to_use = NET_INCOME_HORIZONS
    analysis_mode = "Net Income Growth"
else:
    horizons_to_use = HORIZONS
    analysis_mode = "FCF Growth"

# Run separately for each horizon
for horizon_label, (fcfps_col, price_col) in horizons_to_use.items():
    print(f"\n\nProcessing {horizon_label} horizon ({analysis_mode} mode)...")
    
    # We need the growth columns plus Market_Cap
    df = df_full[["Ticker", "Market_Cap", fcfps_col, price_col]].dropna()

    # Filter to specific index tickers if requested
    index_filter = None
    if args.nasdaq100_only:
        df = df[df["Ticker"].isin(NASDAQ100_TICKERS)]
        index_filter = "Nasdaq-100"
    elif args.dow30_only:
        df = df[df["Ticker"].isin(DOW30_TICKERS)]
        index_filter = "Dow Jones 30"
    elif args.sp500_only:
        df = df[df["Ticker"].isin(SP500_TICKERS)]
        index_filter = "S&P 500"
    
    if index_filter:
        print(f"Filtering to {index_filter} tickers only ({len(df)} observations)")

    # Convert to % points and set up column names
    if args.use_fcf_yield:
        fcf_pct_col = f"{horizon_label}_FCF_yield_pct"
        price_pct_col = f"{horizon_label}_Price_pct"
        df[fcf_pct_col] = df[fcfps_col]  # FCF yield is already in percentage
        df[price_pct_col] = df[price_col] * 100
        x_axis_label = f"{horizon_label} FCF Yield (%)"
    elif args.use_net_income_growth:
        fcf_pct_col = f"{horizon_label}_Net_Income_growth_pct"
        price_pct_col = f"{horizon_label}_Price_pct"
        df[fcf_pct_col] = df[fcfps_col] * 100  # Convert to percentage points
        df[price_pct_col] = df[price_col] * 100
        x_axis_label = f"{horizon_label} Net Income Growth (%)"
    else:
        fcf_pct_col = f"{horizon_label}_FCF_pct"
        price_pct_col = f"{horizon_label}_Price_pct"
        df[fcf_pct_col] = df[fcfps_col] * 100
        df[price_pct_col] = df[price_col] * 100
        x_axis_label = f"{horizon_label} FCFps Growth (%)"

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

    # Create enhanced plot with both OLS and robust lines (skip if --no-plots)
    if not args.no_plots:
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

        ax.set_title(f"{horizon_label} Price Change vs {analysis_mode} by Market Cap Tier" + 
                    (f" ({index_filter} Only)" if index_filter else "") + 
                    f"\n(OLS {'and Robust ' if args.show_robust else ''}Regression with R^2 and RSS)")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        if args.save_plots:
            filename_suffix = f"_{index_filter.lower().replace(' ', '_').replace('-', '_')}" if index_filter else ""
            if args.use_fcf_yield:
                mode_suffix = "_yield"
            elif args.use_net_income_growth:
                mode_suffix = "_netincome_growth"
            else:
                mode_suffix = ""
            plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression{mode_suffix}{filename_suffix}.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression{mode_suffix}{filename_suffix}.png")
        else:
            plt.show()

# Optional: Combined panel showing all horizons (skip if --no-plots)
if args.single_panel and all_horizon_results and not args.no_plots:
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
        
        ax.set_title(f"{horizon_label} Price Change vs {analysis_mode} by Market Cap Tier" + 
                    (f" ({index_filter} Only)" if index_filter else ""))
        if args.use_fcf_yield:
            ax.set_xlabel(f"{horizon_label} FCF Yield (%)")
        elif args.use_net_income_growth:
            ax.set_xlabel(f"{horizon_label} Net Income Growth (%)")
        else:
            ax.set_xlabel(f"{horizon_label} FCFps Growth (%)")
        ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small")
    
    if args.save_plots:
        filename_suffix = f"_{index_filter.lower().replace(' ', '_').replace('-', '_')}" if index_filter else ""
        if args.use_fcf_yield:
            mode_suffix = "_yield"
        elif args.use_net_income_growth:
            mode_suffix = "_netincome_growth"
        else:
            mode_suffix = ""
        plt.savefig(f"all_horizons_market_cap_robust_regression{mode_suffix}{filename_suffix}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved combined plot: all_horizons_market_cap_robust_regression{mode_suffix}{filename_suffix}.png")
    else:
        plt.show()

# Summary statistics across all horizons
print(f"\n{'='*80}")
print(f"SUMMARY: R^2 COMPARISON ACROSS HORIZONS ({analysis_mode})")
print(f"{'='*80}")
print(f"{'Horizon':<10} {'All Samples OLS R^2':<20} {'All Samples Robust R^2':<22} {'Mega-caps OLS R^2':<18} {'Micro-caps OLS R^2':<18}")
print("-" * 80)

for horizon_label in horizons_to_use.keys():
    results = all_horizon_results.get(horizon_label, {})
    all_ols_r2 = results.get("All Samples", {}).get("ols_r2", 0)
    all_robust_r2 = results.get("All Samples", {}).get("robust_r2", 0)
    mega_ols_r2 = results.get("Mega-caps", {}).get("ols_r2", 0)
    micro_ols_r2 = results.get("Micro-caps", {}).get("ols_r2", 0)
    
    print(f"{horizon_label:<10} {all_ols_r2:<20.4f} {all_robust_r2:<22.4f} {mega_ols_r2:<18.4f} {micro_ols_r2:<18.4f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
