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

# Analysis mode configuration
ANALYSIS_MODES = {
    "fcf_growth": {
        "name": "FCF Growth",
        "x_column_template": "{horizon}_FCFps_growth",
        "x_label_template": "{horizon} FCFps Growth (%)",
        "multiply_by_100": True,  # Convert to percentage
        "already_percentage": False
    },
    "fcf_yield": {
        "name": "FCF Yield", 
        "x_column_template": "FCF_yield",  # Same for all horizons
        "x_label_template": "{horizon} FCF Yield (%)",
        "multiply_by_100": False,  # Already in percentage
        "already_percentage": True
    },
    "net_income_growth": {
        "name": "Net Income Growth",
        "x_column_template": "{horizon}_NetIncome_growth", 
        "x_label_template": "{horizon} Net Income Growth (%)",
        "multiply_by_100": True,
        "already_percentage": False
    },
    "volume_growth": {
        "name": "Volume Growth",
        "x_column_template": "{horizon}_Volume_growth",
        "x_label_template": "{horizon} Volume Growth (%)", 
        "multiply_by_100": True,
        "already_percentage": False
    },
    "revenue_growth": {
        "name": "Revenue Growth",
        "x_column_template": "{horizon}_Revenue_growth",
        "x_label_template": "{horizon} Revenue Growth (%)",
        "multiply_by_100": True,
        "already_percentage": False
    }
}

# Time horizons
HORIZONS = ["6M", "1Y", "2Y", "3Y"]

# Load data
df_full = pd.read_csv(CSV, parse_dates=["Report Date"])

def calculate_non_overlapping_net_income_growth(df_full):
    # Sort by Ticker and Report Date to ensure proper time series ordering
    df_full = df_full.sort_values(['Ticker', 'Report Date'])

    # Drop any existing NetIncome growth columns to ensure clean calculation
    existing_ni_cols = [col for col in df_full.columns if 'NetIncome_growth' in col]
    if existing_ni_cols:
        print(f"Removing existing overlapping NetIncome growth columns: {existing_ni_cols}")
        df_full = df_full.drop(columns=existing_ni_cols)

    # Extract quarter information to filter for non-overlapping periods
    df_full["Quarter"] = df_full["Report Date"].dt.quarter

    # Filter to Q1 data only (March quarter-ends) to avoid overlapping periods
    # This ensures we get exactly one observation per year per ticker, eliminating noise from overlapping windows
    df_q1_only = df_full[df_full["Quarter"] == 1].copy().sort_values(['Ticker', 'Report Date'])

    # Calculate growth rates on the filtered Q1-only data for clean non-overlapping periods
    # Note: pct_change(1) on annual Q1 data = 1 year growth, pct_change(2) = 2 year growth, etc.
    df_q1_only["1Y_NetIncome_growth"] = df_q1_only.groupby('Ticker')["Net Income"].pct_change(1)   # 1 year apart (Q1 to Q1)
    df_q1_only["2Y_NetIncome_growth"] = df_q1_only.groupby('Ticker')["Net Income"].pct_change(2)   # 2 years apart  
    df_q1_only["3Y_NetIncome_growth"] = df_q1_only.groupby('Ticker')["Net Income"].pct_change(3)   # 3 years apart

    # Get Q3 data for 6-month comparisons
    df_q3_only = df_full[df_full["Quarter"] == 3].copy().sort_values(['Ticker', 'Report Date'])

    # Calculate 6M growth by comparing Q3 to Q1 of the same year
    df_q1_for_6m = df_q1_only[['Ticker', 'Report Date', 'Net Income']].copy()
    df_q1_for_6m['Year'] = df_q1_for_6m['Report Date'].dt.year
    df_q1_for_6m = df_q1_for_6m.rename(columns={'Net Income': 'Q1_Net_Income'})

    df_q3_for_6m = df_q3_only[['Ticker', 'Report Date', 'Net Income']].copy()
    df_q3_for_6m['Year'] = df_q3_for_6m['Report Date'].dt.year
    df_q3_for_6m = df_q3_for_6m.rename(columns={'Net Income': 'Q3_Net_Income'})

    # Merge Q1 and Q3 data by Ticker and Year to calculate 6M growth
    df_6m_growth = df_q3_for_6m.merge(
        df_q1_for_6m[['Ticker', 'Year', 'Q1_Net_Income']], 
        on=['Ticker', 'Year'], 
        how='left'
    )
    df_6m_growth["6M_NetIncome_growth"] = (df_6m_growth['Q3_Net_Income'] - df_6m_growth['Q1_Net_Income']) / df_6m_growth['Q1_Net_Income']

    # Merge the clean growth rates back to the main dataframe
    growth_6m = df_6m_growth[['Ticker', 'Report Date', '6M_NetIncome_growth']]
    df_full = df_full.merge(growth_6m, on=['Ticker', 'Report Date'], how='left')

    growth_annual = df_q1_only[['Ticker', 'Report Date', '1Y_NetIncome_growth', '2Y_NetIncome_growth', '3Y_NetIncome_growth']]
    df_full = df_full.merge(growth_annual, on=['Ticker', 'Report Date'], how='left')

    print(f"Non-overlapping NetIncome growth calculation complete:")
    print(f"  6M growth: {df_full['6M_NetIncome_growth'].notna().sum()} observations")
    print(f"  1Y growth: {df_full['1Y_NetIncome_growth'].notna().sum()} observations") 
    print(f"  2Y growth: {df_full['2Y_NetIncome_growth'].notna().sum()} observations")
    print(f"  3Y growth: {df_full['3Y_NetIncome_growth'].notna().sum()} observations")

    return df_full

# Call the function to encapsulate the logic
df_full = calculate_non_overlapping_net_income_growth(df_full)

def get_analysis_mode_from_args(args):
    """Determine analysis mode from command line arguments"""
    if args.use_fcf_yield:
        return "fcf_yield"
    elif args.use_net_income_growth:
        return "net_income_growth"
    elif args.use_volume_growth:
        return "volume_growth"
    elif args.use_revenue_growth:
        return "revenue_growth"
    else:
        return "fcf_growth"

def get_column_names(horizon, analysis_mode):
    """Get x and y column names for a given horizon and analysis mode"""
    mode_config = ANALYSIS_MODES[analysis_mode]
    x_col = mode_config["x_column_template"].format(horizon=horizon)
    y_col = f"{horizon}_Price_growth"
    return x_col, y_col

def prepare_data_for_analysis(df, horizon, analysis_mode):
    """Prepare data for analysis by converting to percentages and setting up column names"""
    mode_config = ANALYSIS_MODES[analysis_mode]
    x_col, y_col = get_column_names(horizon, analysis_mode)
    
    # Create percentage column names
    x_pct_col = f"{horizon}_{analysis_mode}_pct"
    y_pct_col = f"{horizon}_Price_pct"
    
    # Convert to percentages
    if mode_config["already_percentage"]:
        df[x_pct_col] = df[x_col]  # Already in percentage
    else:
        df[x_pct_col] = df[x_col] * 100  # Convert to percentage
    
    df[y_pct_col] = df[y_col] * 100  # Always convert price to percentage
    
    # Generate axis label
    x_axis_label = mode_config["x_label_template"].format(horizon=horizon)
    
    return x_pct_col, y_pct_col, x_axis_label

def get_index_filter_info(args):
    """Get index filter information from arguments"""
    if args.nasdaq100_only:
        return NASDAQ100_TICKERS, "Nasdaq-100"
    elif args.dow30_only:
        return DOW30_TICKERS, "Dow Jones 30" 
    elif args.sp500_only:
        return SP500_TICKERS, "S&P 500"
    else:
        return None, None

def get_filename_suffix(analysis_mode, index_filter):
    """Generate filename suffix for saving plots"""
    mode_suffix = f"_{analysis_mode}" if analysis_mode != "fcf_growth" else ""
    index_suffix = f"_{index_filter.lower().replace(' ', '_').replace('-', '_')}" if index_filter else ""
    return mode_suffix + index_suffix

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
parser.add_argument("--use-volume-growth", action="store_true",
                    help="Use Volume Growth rates as the x-axis instead of FCF growth rates")
parser.add_argument("--use-revenue-growth", action="store_true",
                    help="Use Revenue Growth rates as the x-axis instead of FCF growth rates")
parser.add_argument("--by-year-windows", action="store_true",
                    help="Produce separate graphs for each sliding year window (e.g., 2019-2021, 2020-2022, etc.)")
parser.add_argument("--window-timeframe", choices=["6M", "1Y", "2Y", "3Y"], default="1Y",
                    help="Time frame to use when --by-year-windows is enabled (default: 1Y)")
args = parser.parse_args()

# Validation for year windows mode
if args.by_year_windows:
    print(f"Year windows mode enabled - analyzing {args.window_timeframe} horizon across sliding year windows")
    if args.single_panel:
        print("Warning: --single-panel will be ignored in year windows mode")
        args.single_panel = False

# Calculate FCF Yield if using yield mode
if args.use_fcf_yield:
    # Calculate FCF Yield as FCF / Market_Cap (expressed as percentage)
    # Using Market Cap as proxy for EV since EV data is not available
    df_full["FCF_yield"] = (df_full["FCF"] / df_full["Market_Cap"]) * 100

# -----------------------------------------------------------------------------
# Enhanced helper to clip & regress & report with R^2 and RSS
# -----------------------------------------------------------------------------
def enhanced_regression_analysis(x, y, p_low=0.2, p_high=99.8):
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

# Determine analysis mode and configuration
analysis_mode_key = get_analysis_mode_from_args(args)
analysis_mode_config = ANALYSIS_MODES[analysis_mode_key]
analysis_mode_name = analysis_mode_config["name"]

# Get index filter information
index_tickers, index_filter = get_index_filter_info(args)

# Choose horizons to analyze
horizons_to_analyze = HORIZONS.copy()

# Handle year windows mode
if args.by_year_windows:
    # Use only the specified timeframe
    horizons_to_analyze = [args.window_timeframe]
    
    # Get unique years from the dataset for sliding windows
    df_full['Year'] = df_full['Report Date'].dt.year
    available_years = sorted(df_full['Year'].unique())
    
    # Determine window size based on timeframe
    if args.window_timeframe == "6M":
        window_size = 1  # 6M uses same year data
    elif args.window_timeframe == "1Y":
        window_size = 2  # 1Y needs 2 years (start year + 1)
    elif args.window_timeframe == "2Y":
        window_size = 3  # 2Y needs 3 years (start year + 2)
    elif args.window_timeframe == "3Y":
        window_size = 4  # 3Y needs 4 years (start year + 3)
    
    # Create sliding windows
    year_windows = []
    for i in range(len(available_years) - window_size + 1):
        start_year = available_years[i]
        end_year = available_years[i + window_size - 1]
        year_windows.append((start_year, end_year))
    
    print(f"Available year windows for {args.window_timeframe} analysis: {year_windows}")

def analyze_horizon_data(df, horizon, analysis_mode_key, index_tickers, index_filter):
    """Analyze data for a specific horizon and return results"""
    x_col, y_col = get_column_names(horizon, analysis_mode_key)
    
    # Filter data
    df_filtered = df[["Ticker", "Market_Cap", x_col, y_col]].dropna()
    
    # Apply index filter if specified
    if index_tickers is not None:
        df_filtered = df_filtered[df_filtered["Ticker"].isin(index_tickers)]
        print(f"Filtering to {index_filter} tickers only ({len(df_filtered)} observations)")
    
    if len(df_filtered) < 10:
        print(f"Insufficient data ({len(df_filtered)} observations)")
        return None, None, None
    
    # Prepare data for analysis
    x_pct_col, y_pct_col, x_axis_label = prepare_data_for_analysis(df_filtered, horizon, analysis_mode_key)
    
    # Define cap tiers
    largest_market_cap_stocks = df_filtered["Market_Cap"].quantile(EXTREME_COMPANIES_PERCENTS / 100)
    smallest_market_cap_stocks = df_filtered["Market_Cap"].quantile(1 - (EXTREME_COMPANIES_PERCENTS / 100))

    tiers = {
        "All Samples": df_filtered.index,
        "Mega-caps": df_filtered[df_filtered["Market_Cap"] >= smallest_market_cap_stocks].index,
        "Micro-caps": df_filtered[df_filtered["Market_Cap"] <= largest_market_cap_stocks].index,
        "Mid-caps": df_filtered[(df_filtered["Market_Cap"] > largest_market_cap_stocks) & (df_filtered["Market_Cap"] < smallest_market_cap_stocks)].index,
    }

    # Run enhanced analysis for all tiers
    results = {}
    for name, idx in tiers.items():
        sub = df_filtered.loc[idx]
        if len(sub) < 5:  # Skip if too few observations
            continue
            
        x_vals = sub[x_pct_col].values
        y_vals = sub[y_pct_col].values
        
        if name == "Micro-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                           p_low=P_LOWEST_CAP_CLIPPING_LOW, 
                                           p_high=P_LOWEST_CAP_CLIPPING_HIGH)
        elif name == "Mega-caps" or name == "Mid-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                            p_low=0.9, 
                                            p_high=99.1)
        else:
            r = enhanced_regression_analysis(x_vals, y_vals)

        if r:
            results[name] = r

    return results, x_axis_label, df_filtered

# Run analysis based on mode
if args.by_year_windows:
    # Year windows mode - analyze each year window separately
    for start_year, end_year in year_windows:
        print(f"\n{'='*80}")
        print(f"ANALYZING YEAR WINDOW: {start_year}-{end_year}")
        print(f"{'='*80}")
        
        # Filter data to the specific year window
        df_window = df_full[
            (df_full['Year'] >= start_year) & 
            (df_full['Year'] <= end_year)
        ].copy()
        
        # Run analysis for the single timeframe on this year window
        for horizon_label in horizons_to_analyze:
            print(f"\nProcessing {horizon_label} horizon for {start_year}-{end_year} ({analysis_mode_name} mode)...")
            
            # Use the new generic analysis function
            results, x_axis_label, df_filtered = analyze_horizon_data(
                df_window, horizon_label, analysis_mode_key, index_tickers, index_filter
            )
            
            if results is None:
                continue

            # Print formatted results table
            format_results_table(results, f"{horizon_label} ({start_year}-{end_year})")

            # Create plot for this year window (skip if --no-plots)
            if not args.no_plots and results:
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
                            label=f"{name} OLS (R^2={r['ols_r2']:.3f}, ρ={r['correlations']['pearson']:.3f})")
                    
                    # Robust fit line (if enabled)
                    if args.show_robust:
                        robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
                        ax.plot(fit_x, robust_fit_y, color=colors[name], lw=2, linestyle='--',
                                label=f"{name} Robust (R^2={r['robust_r2']:.3f})")

                ax.set_title(f"{horizon_label} Price Change vs {analysis_mode_name} by Market Cap Tier ({start_year}-{end_year})" + 
                            (f" ({index_filter} Only)" if index_filter else "") + 
                            f"\n(OLS {'and Robust ' if args.show_robust else ''}Regression with R^2 and RSS)")
                ax.set_xlabel(x_axis_label)
                ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(alpha=0.3)
                plt.tight_layout()
                
                if args.save_plots:
                    filename_suffix = get_filename_suffix(analysis_mode_key, index_filter)
                    plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression{filename_suffix}_{start_year}_{end_year}.png", 
                               dpi=300, bbox_inches='tight')
                    print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression{filename_suffix}_{start_year}_{end_year}.png")
                else:
                    plt.show()
else:
    # Regular mode - analyze all horizons normally
    for horizon_label in horizons_to_analyze:
        print(f"\n\nProcessing {horizon_label} horizon ({analysis_mode_name} mode)...")
        
        # Use the new generic analysis function
        results, x_axis_label, df_filtered = analyze_horizon_data(
            df_full, horizon_label, analysis_mode_key, index_tickers, index_filter
        )
        
        if results is None:
            continue

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
                        label=f"{name} OLS (R^2={r['ols_r2']:.3f}, ρ={r['correlations']['pearson']:.3f})")
                
                # Robust fit line (if enabled)
                if args.show_robust:
                    robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
                    ax.plot(fit_x, robust_fit_y, color=colors[name], lw=2, linestyle='--',
                            label=f"{name} Robust (R^2={r['robust_r2']:.3f})")

            ax.set_title(f"{horizon_label} Price Change vs {analysis_mode_name} by Market Cap Tier" + 
                        (f" ({index_filter} Only)" if index_filter else "") + 
                        f"\n(OLS {'and Robust ' if args.show_robust else ''}Regression with R^2 and RSS)")
            ax.set_xlabel(x_axis_label)
            ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            if args.save_plots:
                filename_suffix = get_filename_suffix(analysis_mode_key, index_filter)
                plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression{filename_suffix}.png", 
                           dpi=300, bbox_inches='tight')
                print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression{filename_suffix}.png")
            else:
                plt.show()

# Optional: Combined panel showing all horizons (skip if --no-plots and not in year windows mode)
if args.single_panel and all_horizon_results and not args.no_plots and not args.by_year_windows:
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    colors = {"All Samples": "black", "Mega-caps": "blue", "Mid-caps": "green", "Micro-caps": "orange"}
    
    for i, horizon_label in enumerate(HORIZONS):
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
                    label=f"{name} OLS (R^2={r['ols_r2']:.3f}, ρ={r['correlations']['pearson']:.3f})")
            
            if args.show_robust:
                robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
                ax.plot(fit_x, robust_fit_y, color=colors.get(name, "black"), 
                       lw=2, linestyle='--', label=f"{name} Robust")
        
        ax.set_title(f"{horizon_label} Price Change vs {analysis_mode_name} by Market Cap Tier" + 
                    (f" ({index_filter} Only)" if index_filter else ""))
        x_axis_label = ANALYSIS_MODES[analysis_mode_key]["x_label_template"].format(horizon=horizon_label)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(f"{horizon_label} Forward Price Change (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize="x-small")
    
    if args.save_plots:
        filename_suffix = get_filename_suffix(analysis_mode_key, index_filter)
        plt.savefig(f"all_horizons_market_cap_robust_regression{filename_suffix}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved combined plot: all_horizons_market_cap_robust_regression{filename_suffix}.png")
    else:
        plt.show()

# Summary statistics across all horizons (skip in year windows mode)
if not args.by_year_windows:
    print(f"\n{'='*80}")
    print(f"SUMMARY: R^2 COMPARISON ACROSS HORIZONS ({analysis_mode_name})")
    print(f"{'='*80}")
    print(f"{'Horizon':<10} {'All Samples OLS R^2':<20} {'All Samples Robust R^2':<22} {'Mega-caps OLS R^2':<18} {'Micro-caps OLS R^2':<18}")
    print("-" * 80)

    for horizon_label in HORIZONS:
        results = all_horizon_results.get(horizon_label, {})
        all_ols_r2 = results.get("All Samples", {}).get("ols_r2", 0)
        all_robust_r2 = results.get("All Samples", {}).get("robust_r2", 0)
        mega_ols_r2 = results.get("Mega-caps", {}).get("ols_r2", 0)
        micro_ols_r2 = results.get("Micro-caps", {}).get("ols_r2", 0)
        
        print(f"{horizon_label:<10} {all_ols_r2:<20.4f} {all_robust_r2:<22.4f} {mega_ols_r2:<18.4f} {micro_ols_r2:<18.4f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
