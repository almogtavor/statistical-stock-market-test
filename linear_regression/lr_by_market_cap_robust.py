#!/usr/bin/env python3
"""

Linear Regression analysis that splits into cap tiers (top10%, mid80%, bottom10%, all)
Price change vs FCFps change or Revenue change analysis for each segment, including:
- LS regression with R^2, RSS
- Resistance line (robust regression using terciles)
- Comprehensive statistical metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import warnings
import math
from reject_null_hypothesis import regression_analysis
warnings.filterwarnings('ignore')

EXTREME_COMPANIES_PERCENTS = 10

# Constants
CSV = "../fcf_dataset.csv"
HORIZONS = ["6M", "1Y", "2Y", "3Y"]

# Index ticker lists
INDEX_TICKERS = {
    "Nasdaq-100": ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS'],
    "Dow Jones 30": ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT'],
    "S&P 500": ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
}

# Analysis mode configuration
ANALYSIS_MODES = {
    "fcf_growth": {"name": "FCF Growth", "x_column_template": "{horizon}_FCFps_growth", "x_label_template": "{horizon} FCFps Growth (%)", "already_percentage": False},
    "fcf_yield": {"name": "FCF Yield", "x_column_template": "FCF_yield", "x_label_template": "{horizon} FCF Yield (%)", "already_percentage": True},
    "net_income_growth": {"name": "Net Income Growth", "x_column_template": "{horizon}_NetIncome_growth", "x_label_template": "{horizon} Net Income Growth (%)", "already_percentage": False},
    "volume_growth": {"name": "Volume Growth", "x_column_template": "{horizon}_Volume_growth", "x_label_template": "{horizon} Volume Growth (%)", "already_percentage": False},
    "revenue_growth": {"name": "Revenue Growth", "x_column_template": "{horizon}_Revenue_growth", "x_label_template": "{horizon} Revenue Growth (%)", "already_percentage": False}
}

# Load and process data
df_full = pd.read_csv(CSV, parse_dates=["Report Date"])

def calculate_non_overlapping_net_income_growth(df):
    """Calculate non-overlapping net income growth rates"""
    df = df.sort_values(['Ticker', 'Report Date'])
    
    # Remove existing columns if they exist
    existing_cls = [col for col in df.columns if 'NetIncome_growth' in col]
    if existing_cls:
        df = df.drop(columns=existing_cls)
    
    df["Quarter"] = df["Report Date"].dt.quarter
    
    # Q1 data for annual calculations
    df_q1 = df[df["Quarter"] == 1].copy().sort_values(['Ticker', 'Report Date'])
    for i, horizon in enumerate(["1Y", "2Y", "3Y"], 1):
        df_q1[f"{horizon}_NetIncome_growth"] = df_q1.groupby('Ticker')["Net Income"].pct_change(i)
    
    # Q3 data for 6M calculations
    df_q3 = df[df["Quarter"] == 3].copy()
    df_q1_6m = df_q1[['Ticker', 'Report Date', 'Net Income']].copy()
    df_q1_6m['Year'] = df_q1_6m['Report Date'].dt.year
    df_q3['Year'] = df_q3['Report Date'].dt.year
    
    df_6m = df_q3.merge(df_q1_6m, on=['Ticker', 'Year'], suffixes=('_Q3', '_Q1'), how='left')
    df_6m["6M_NetIncome_growth"] = (df_6m['Net Income_Q3'] - df_6m['Net Income_Q1']) / df_6m['Net Income_Q1']
    
    # For 6M data, use the Q3 Report Date (which has suffix _Q3)
    df_6m['Report Date'] = df_6m['Report Date_Q3']
    
    # Merge back
    for data, cls in [(df_6m, ['6M_NetIncome_growth']), (df_q1, ['1Y_NetIncome_growth', '2Y_NetIncome_growth', '3Y_NetIncome_growth'])]:
        merge_cls = ['Ticker', 'Report Date'] + cls
        df = df.merge(data[merge_cls], on=['Ticker', 'Report Date'], how='left')
    
    return df

df_full = calculate_non_overlapping_net_income_growth(df_full)

def get_analysis_mode_from_args(args):
    """Determine analysis mode from command line arguments"""
    modes = [("use_fcf_yield", "fcf_yield"), ("use_net_income_growth", "net_income_growth"), 
             ("use_volume_growth", "volume_growth"), ("use_revenue_growth", "revenue_growth")]
    return next((mode for attr, mode in modes if getattr(args, attr)), "fcf_growth")

def get_column_names(horizon, analysis_mode):
    """Get x and y column names for a given horizon and analysis mode"""
    return ANALYSIS_MODES[analysis_mode]["x_column_template"].format(horizon=horizon), f"{horizon}_Price_growth"

def prepare_data_for_analysis(df, horizon, analysis_mode, use_log_price_change=False, log_x_axis=False):
    """Prepare data for analysis by converting to percentages and setting up column names"""
    mode_config = ANALYSIS_MODES[analysis_mode]
    x_col, y_col = get_column_names(horizon, analysis_mode)
    
    # Create column names
    x_pct_col = f"{horizon}_{analysis_mode}_{'log' if log_x_axis else 'pct'}"
    y_pct_col = f"{horizon}_Price_{'log' if use_log_price_change else 'pct'}"
    
    # Helper function for safe log transformation
    def safe_log(x, is_percentage=False):
        if pd.isna(x) or np.isinf(x) or isinstance(x, str):
            return np.nan
        return math.log(abs(x) + 0.01) if is_percentage else math.log(1 + x) if (1 + x) > 0 else np.nan
    
    # Convert x-axis
    if log_x_axis:
        df[x_pct_col] = df[x_col].apply(lambda x: safe_log(x, mode_config["already_percentage"]))
    else:
        df[x_pct_col] = df[x_col] if mode_config["already_percentage"] else df[x_col] * 100
    
    # Convert y-axis
    if use_log_price_change:
        df[y_pct_col] = df[y_col].apply(lambda x: safe_log(x, False))
    else:
        df[y_pct_col] = df[y_col] * 100
    
    # Generate axis label
    x_axis_label = mode_config["x_label_template"].format(horizon=horizon)
    if log_x_axis:
        x_axis_label = f"Log {x_axis_label.replace(' (%)', '') if mode_config['already_percentage'] else x_axis_label}"
    
    return x_pct_col, y_pct_col, x_axis_label

def get_index_filter_info(args):
    """Get index filter information from arguments"""
    index_map = {
        "nasdaq100_only": "Nasdaq-100",
        "dow30_only": "Dow Jones 30", 
        "sp500_only": "S&P 500"
    }
    for attr, name in index_map.items():
        if getattr(args, attr):
            return INDEX_TICKERS[name], name
    return None, None

def get_filename_suffix(analysis_mode, index_filter, use_log_price_change=False, log_x_axis=False):
    """Generate filename suffix for saving plots"""
    parts = []
    if analysis_mode != "fcf_growth":
        parts.append(analysis_mode)
    if index_filter:
        parts.append(index_filter.lower().replace(' ', '_').replace('-', '_'))
    if use_log_price_change:
        parts.append("log_price")
    if log_x_axis:
        parts.append("log_x")
    return "_" + "_".join(parts) if parts else ""

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--single-panel", action="store_true", help="Overlay all horizons' All Samples results on one panel")
parser.add_argument("--show-robust", action="store_true", default=True, help="Show resistance line alongside LS")
parser.add_argument("--save-plots", action="store_true", help="Save plots to files instead of showing them")
parser.add_argument("--no-plots", action="store_true", help="Skip all plotting and show only statistical results")
parser.add_argument("--nasdaq100-only", action="store_true", help="Restrict analysis to Nasdaq-100 tickers only")
parser.add_argument("--dow30-only", action="store_true", help="Restrict analysis to Dow Jones 30 tickers only")
parser.add_argument("--sp500-only", action="store_true", help="Restrict analysis to S&P 500 tickers only")
parser.add_argument("--use-fcf-yield", action="store_true", help="Use FCF Yield instead of FCF growth rates")
parser.add_argument("--use-net-income-growth", action="store_true", help="Use Net Income Growth rates")
parser.add_argument("--use-volume-growth", action="store_true", help="Use Volume Growth rates")
parser.add_argument("--use-revenue-growth", action="store_true", help="Use Revenue Growth rates")
parser.add_argument("--by-year-windows", action="store_true", help="Produce separate graphs for each sliding year window")
parser.add_argument("--window-timeframe", choices=["6M", "1Y", "2Y", "3Y"], default="1Y", help="Time frame for year windows mode")
parser.add_argument("--use-log-price-change", action="store_true", help="Use log price change on the y-axis")
parser.add_argument("--log-x-axis", action="store_true", help="Apply log transformation to the x-axis values")
args = parser.parse_args()

# Validation and setup
if args.by_year_windows and args.single_panel:
    print("Warning: --single-panel ignored in year windows mode")
    args.single_panel = False

if args.use_fcf_yield:
    df_full["FCF_yield"] = (df_full["FCF"] / df_full["Market_Cap"]) * 100

# -----------------------------------------------------------------------------
# Enhanced helper to clip & regress & report with R^2 and RSS
# -----------------------------------------------------------------------------
def enhanced_regression_analysis(x, y, p_low=0.9, p_high=99.1):
    """
    Perform both LS and robust regression with comprehensive metrics
    """
    # Filter out infinite and NaN values first
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    
    if len(x) < 5:
        return None
    
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
    
    # LS Regression (manual calculation for consistency with original)
    x_mean, y_mean = x_clipped.mean(), y_clipped.mean()
    Sxx = np.sum((x_clipped - x_mean) ** 2)
    Sxy = np.sum((x_clipped - x_mean) * (y_clipped - y_mean))
    
    if Sxx == 0:
        return None
        
    ls_slope = Sxy / Sxx
    ls_intercept = y_mean - ls_slope * x_mean
    
    # Predictions and residuals
    y_pred_ls = ls_intercept + ls_slope * x_clipped
    residuals_ls = y_clipped - y_pred_ls
    
    # RSS (Residual Sum of Squares)
    rss_ls = np.sum(residuals_ls ** 2)
    
    # R^2 calculation
    tss = np.sum((y_clipped - y_mean) ** 2)  # Total Sum of Squares
    r2_ls = 1 - (rss_ls / tss) if tss > 0 else 0
    
    # Standard error and statistical tests
    residual_var = rss_ls / (n - 2)
    SE = np.sqrt(residual_var / Sxx)
    t_stat = ls_slope / SE if SE > 0 else 0
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 2) if SE > 0 else 1
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha /2, df=n - 2)
    ci = (ls_slope - t_crit * SE, ls_slope + t_crit * SE) if SE > 0 else (ls_slope, ls_slope)
    reject_H0 = abs(t_stat) > t_crit if SE > 0 else False
    
    # Resistance Line (Robust Regression using terciles)
    try:
        # Sort data by X values to find terciles
        sorted_indices = np.argsort(x_clipped)
        n = len(x_clipped)
        
        # Divide into thirds (terciles)
        tercile_size = n // 3
        
        # Lower tercile (bottom 1/3)
        lower_indices = sorted_indices[:tercile_size]
        X_L = np.median(x_clipped[lower_indices])  # Median of X in lower tercile
        Y_L = np.median(y_clipped[lower_indices])  # Median of Y in lower tercile
        
        # Upper tercile (top 1/3)
        upper_indices = sorted_indices[-tercile_size:]
        X_H = np.median(x_clipped[upper_indices])  # Median of X in upper tercile
        Y_H = np.median(y_clipped[upper_indices])  # Median of Y in upper tercile
        
        # Calculate resistance line slope: b_RL = (Y_H - Y_L) / (X_H - X_L)
        if X_H != X_L:
            robust_slope = (Y_H - Y_L) / (X_H - X_L)
            
            # Calculate residuals from the line: r_i* = Y_i - b_RL * X_i
            residuals_from_slope = y_clipped - robust_slope * x_clipped
            
            # Intercept is the median of residuals: a_RL = med(r_i*)
            robust_intercept = np.median(residuals_from_slope)
            
            # Calculate predictions and final residuals
            y_pred_robust = robust_intercept + robust_slope * x_clipped
            residuals_robust = y_clipped - y_pred_robust
            rss_robust = np.sum(residuals_robust ** 2)
            
            # Calculate R² using the mathematical formula: r²xy = 1 - (RSS/TSS)
            r2_robust = 1 - (rss_robust / tss) if tss > 0 else 0
        else:
            # Fallback if X values are identical in terciles
            robust_slope = ls_slope
            robust_intercept = ls_intercept
            rss_robust = rss_ls
            r2_robust = r2_ls
        
    except Exception as e:
        print(f"Resistance line calculation failed: {e}")
        robust_slope = ls_slope
        robust_intercept = ls_intercept
        rss_robust = rss_ls
        r2_robust = r2_ls
    pearson_corr, _ = stats.pearsonr(x_clipped, y_clipped)
    
    return {
        "n": n,
        # LS results
        "ls_intercept": ls_intercept,
        "ls_slope": ls_slope,
        "ls_SE": SE,
        "ls_t": t_stat,
        "ls_p_value": p_value,
        "ls_CI": ci,
        "ls_r2": r2_ls,
        "ls_rss": rss_ls,
        "ls_t_crit": t_crit,
        "ls_reject_H0": reject_H0,
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
        }
    }


def format_results_table(results_dict, horizon_label):
    """Print a nicely formatted table of results"""
    print(f"\n{'='*80}\nREGRESSION RESULTS FOR {horizon_label}\n{'='*80}")

    headers = ["Tier", "N", "LS b_LS", "R²", "RSS", "Resistance b_RL", "Resistance RSS", "p-value", "t-stat", "t-crit", "SE", "Reject H0", "Pearson Corr"]
    print(f"{headers[0]:<12} {headers[1]:<6} {headers[2]:<10} {headers[3]:<8} {headers[4]:<12} {headers[5]:<10} {headers[6]:<8} {headers[7]:<12} {headers[8]:<10} {headers[9]:<8} {headers[10]:<8} {headers[11]:<10} {headers[12]:<14}")
    print("-" * 150)

    for name, r in results_dict.items():
        if r:
            print(f"{name:<12} {r['n']:<6} {r['ls_slope']:<10.4f} {r['ls_r2']:<8.3f} {r['ls_rss']:<12.1f} "
                  f"{r['robust_slope']:<10.4f} {r['robust_r2']:<8.3f} {r['robust_rss']:<12.1f} {r['ls_p_value']:<10.3g} "
                  f"{r['ls_t']:<8.3f} {r['ls_t_crit']:<8.3f} {r['ls_SE']:<10.3g} {str(r['ls_reject_H0']):<10} "
                  f"{r['correlations']['pearson']:<14.3f}")


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

def create_regression_plot(results, horizon_label, analysis_mode_name, x_axis_label, args, index_filter=None, year_range=None):
    """Create a regression plot with scatter points and fit lines"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use black color for index-specific analysis, otherwise use tier-specific colors
    if index_filter:
        colors = {"All Samples": "black"}
    else:
        colors = {"All Samples": "green", "Mega-caps": "blue", "Mid-caps": "black", "Micro-caps": "orange"}
    
    for name, r in results.items():
        if not r:
            continue
        
    for name, r in results.items():
        if not r:
            continue
        
        # Skip "All Samples" scatter plot unless we're analyzing a specific index
        if not (name == "All Samples" and not index_filter):
            # Scatter plot
            color = colors.get(name)
            ax.scatter(r["x_clipped"], r["y_clipped"], s=15, alpha=0.4, 
                      color=color, label=f"{name} data (n={r['n']})")
        
        # Always plot regression lines (including for "All Samples")
        color = colors.get(name)
        fit_x = np.linspace(*r["xlim"], 200)
        ls_fit_y = r["ls_intercept"] + r["ls_slope"] * fit_x
        ax.plot(fit_x, ls_fit_y, color=color, lw=2, 
                label=f"{name} LS (b_LS={r['ls_slope']:.3f}, R²={r['ls_r2']:.3f}, ρ={r['correlations']['pearson']:.3f})")

        if args.show_robust:
            robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
            ax.plot(fit_x, robust_fit_y, color=color, lw=2, linestyle='--',
                    label=f"{name} Resistance (b_RL={r['robust_slope']:.3f})")

    # Set labels and title    
    title = f"{horizon_label} Price Change vs {analysis_mode_name}"
    if year_range:
        title += f" ({year_range})"
    if index_filter:
        title += f" ({index_filter} Only)"
    else:
        title += " by Market Cap Tier"

    ax.set_title(title)
    ax.set_xlabel(x_axis_label)
    y_label = f"{horizon_label} Forward Log Price Change" if args.use_log_price_change else f"{horizon_label} Forward Price Change (%)"
    ax.set_ylabel(y_label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_single_subplot(ax, results, horizon_label, analysis_mode_name, x_axis_label, args, index_filter):
    """Plot regression data on a single subplot axis"""
    
    # Use black color for index-specific analysis, otherwise use tier-specific colors
    if index_filter:
        colors = {"All Samples": "black", "Mega-caps": "black", "Mid-caps": "black", "Micro-caps": "black"}
    else:
        colors = {"All Samples": "green", "Mega-caps": "blue", "Mid-caps": "black", "Micro-caps": "orange"}
    
    for name, r in results.items():
        if not r:
            continue
        
    for name, r in results.items():
        if not r:
            continue
        
        # Skip "All Samples" scatter plot unless we're analyzing a specific index
        if not (name == "All Samples" and not index_filter):
            # Use proper color for each tier
            color = colors.get(name, "black")
            ax.scatter(r["x_clipped"], r["y_clipped"], s=10, alpha=0.3, 
                      color=color, label=f"{name} (n={r['n']})")
        
        # Always plot regression lines (including for "All Samples")
        color = colors.get(name, "black")
        fit_x = np.linspace(*r["xlim"], 100)
        ls_fit_y = r["ls_intercept"] + r["ls_slope"] * fit_x
        ax.plot(fit_x, ls_fit_y, color=color, lw=2, 
               label=f"{name} LS (b_LS={r['ls_slope']:.3f}, R²={r['ls_r2']:.3f})")
        
        # Robust fit line (if enabled)
        if args.show_robust:
            robust_fit_y = r["robust_intercept"] + r["robust_slope"] * fit_x
            ax.plot(fit_x, robust_fit_y, color=color, 
                   lw=2, linestyle='--', label=f"{name} Resistance (b_RL={r['robust_slope']:.3f})")

    ax.set_title(f"{horizon_label} vs {analysis_mode_name}" + 
                (f" ({index_filter})" if index_filter else ""))
    ax.set_xlabel(x_axis_label)
    y_label = f"{horizon_label} Log Price Change" if args.use_log_price_change else f"{horizon_label} Price Change (%)"
    ax.set_ylabel(y_label)
    ax.legend(fontsize="x-small")
    ax.grid(alpha=0.3)

def analyze_horizon_data(df, horizon, analysis_mode_key, index_tickers, index_filter, use_log_price_change=False, log_x_axis=False):
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
    x_pct_col, y_pct_col, x_axis_label = prepare_data_for_analysis(df_filtered, horizon, analysis_mode_key, use_log_price_change, log_x_axis)
    
    # Define cap tiers - if analyzing a specific index, only use "All Samples"
    if index_filter:
        tiers = {
            "All Samples": df_filtered.index,
        }
    else:
        # Define cap tiers for general analysis
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
        
        # Extract values and filter out NaN values after transformation
        valid_mask = sub[x_pct_col].notna() & sub[y_pct_col].notna()
        sub_clean = sub[valid_mask]
        
        if len(sub_clean) < 5:  # Skip if too few valid observations after filtering NaN
            continue
            
        x_vals = sub_clean[x_pct_col].values
        y_vals = sub_clean[y_pct_col].values
        print(name)
        
        # Use appropriate clipping based on tier and whether we're analyzing an index
        if index_filter:
            # For index analysis, use standard clipping for all samples
            r = enhanced_regression_analysis(x_vals, y_vals, p_low=0.5, p_high=99.5)
        elif name == "Micro-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                           p_low=3, 
                                           p_high=97)
        elif name == "Mega-caps" or name == "Mid-caps":
            r = enhanced_regression_analysis(x_vals, y_vals,
                                            p_low=1.5, 
                                            p_high=98.5)
        else:
            r = enhanced_regression_analysis(x_vals, y_vals,
                                            p_low=1.5, 
                                            p_high=98.5)

        if r:
            results[name] = r
        else:
            print(f"  Failed to process {name} - regression analysis returned None")

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
            # Use the new generic analysis function
            results, x_axis_label, df_filtered = analyze_horizon_data(
                df_window, horizon_label, analysis_mode_key, index_tickers, index_filter, args.use_log_price_change, args.log_x_axis
            )
            
            if results is None:
                continue

            # Print formatted results table
            format_results_table(results, f"{horizon_label} ({start_year}-{end_year})")

            # Create plot for this year window (skip if --no-plots)
            if not args.no_plots and results:
                fig = create_regression_plot(results, horizon_label, analysis_mode_name, x_axis_label, args, 
                                           index_filter, f"{start_year}-{end_year}")
                
                if args.save_plots:
                    filename_suffix = get_filename_suffix(analysis_mode_key, index_filter, args.use_log_price_change, args.log_x_axis)
                    plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression{filename_suffix}_{start_year}_{end_year}.png", 
                               dpi=300, bbox_inches='tight')
                    print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression{filename_suffix}_{start_year}_{end_year}.png")
                else:
                    plt.show()
else:
    # Regular mode - analyze all horizons normally
    for horizon_label in horizons_to_analyze:        
        # Use the new generic analysis function
        results, x_axis_label, df_filtered = analyze_horizon_data(
            df_full, horizon_label, analysis_mode_key, index_tickers, index_filter, args.use_log_price_change, args.log_x_axis
        )
        
        if results is None:
            continue

        # Print formatted results table
        format_results_table(results, horizon_label)

        # Store results for plotting
        if "All Samples" in results:
            all_samples_results[horizon_label] = results["All Samples"]
        all_horizon_results[horizon_label] = results

        # Create enhanced plot with both LS and robust lines (skip if --no-plots)
        if not args.no_plots:
            fig = create_regression_plot(results, horizon_label, analysis_mode_name, x_axis_label, args, index_filter)
            
            if args.save_plots:
                filename_suffix = get_filename_suffix(analysis_mode_key, index_filter, args.use_log_price_change, args.log_x_axis)
                plt.savefig(f"{horizon_label.lower()}_market_cap_robust_regression{filename_suffix}.png", 
                           dpi=300, bbox_inches='tight')
                print(f"Saved plot: {horizon_label.lower()}_market_cap_robust_regression{filename_suffix}.png")
            else:
                plt.show()

# Optional: Combined panel showing all horizons (skip if --no-plots and not in year windows mode)
if args.single_panel and all_horizon_results and not args.no_plots and not args.by_year_windows:
    fig, axs = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    
    # Use black color for index-specific analysis, otherwise use tier-specific colors
    if index_filter:
        colors = {"All Samples": "black", "Mega-caps": "black", "Mid-caps": "black", "Micro-caps": "black"}
    else:
        colors = {"All Samples": "green", "Mega-caps": "blue", "Mid-caps": "black", "Micro-caps": "orange"}
    
    for i, horizon_label in enumerate(HORIZONS):
        row, col = divmod(i, 2)
        ax = axs[row, col]
        results = all_horizon_results.get(horizon_label, {})
        
        x_axis_label = ANALYSIS_MODES[analysis_mode_key]["x_label_template"].format(horizon=horizon_label)
        plot_single_subplot(ax, results, horizon_label, analysis_mode_name, x_axis_label, args, index_filter)
    
    if args.save_plots:
        filename_suffix = get_filename_suffix(analysis_mode_key, index_filter, args.use_log_price_change, args.log_x_axis)
        plt.savefig(f"all_horizons_market_cap_robust_regression{filename_suffix}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Saved combined plot: all_horizons_market_cap_robust_regression{filename_suffix}.png")
    else:
        plt.show()

# Summary statistics across all horizons (skip in year windows mode)
if not args.by_year_windows:
    print(f"\n{'='*80}")
    print(f"SUMMARY: R^2 COMPARISON ACROSS HORIZONS ({analysis_mode_name})")
    if args.use_log_price_change:
        print("Using log price change on y-axis")
    print(f"{'='*80}")
    print(f"{'Horizon':<10} {'All Samples LS R^2':<20} {'All Samples Resistance R^2':<22} {'Mega-caps LS R^2':<18} {'Micro-caps LS R^2':<18}")
    print("-" * 80)

    for horizon_label in HORIZONS:
        results = all_horizon_results.get(horizon_label, {})
        all_ls_r2 = results.get("All Samples", {}).get("ls_r2", 0)
        all_robust_r2 = results.get("All Samples", {}).get("robust_r2", 0)
        mega_ls_r2 = results.get("Mega-caps", {}).get("ls_r2", 0)
        micro_ls_r2 = results.get("Micro-caps", {}).get("ls_r2", 0)
        
        print(f"{horizon_label:<10} {all_ls_r2:<20.4f} {all_robust_r2:<22.4f} {mega_ls_r2:<18.4f} {micro_ls_r2:<18.4f}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
