#!/usr/bin/env python3
"""
Stock Price Prediction Script

Identifies companies expected to have the best positive stock changes
based on recent financial performance (Q2 2024 to present).
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Constants
CSV = "../fcf_dataset.csv"

# Index ticker lists
INDEX_TICKERS = {
    "Nasdaq-100": ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS'],
    "Dow Jones 30": ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT'],
    "S&P 500": ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
}

def calculate_regression_coefficients(x, y, p_low=1.5, p_high=98.5):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    
    if len(x) < 10:
        return None, None, 0, 0
    
    x_clip_min, x_clip_max = np.percentile(x, [p_low, p_high])
    y_clip_min, y_clip_max = np.percentile(y, [p_low, p_high])
    mask = (x >= x_clip_min) & (x <= x_clip_max) & (y >= y_clip_min) & (y <= y_clip_max)
    x_clipped, y_clipped = x[mask], y[mask]
    
    if len(x_clipped) < 10:
        return None, None, 0, 0
    
    x_mean, y_mean = x_clipped.mean(), y_clipped.mean()
    Sxx = np.sum((x_clipped - x_mean) ** 2)
    Sxy = np.sum((x_clipped - x_mean) * (y_clipped - y_mean))
    
    if Sxx == 0:
        return None, None, 0, 0
        
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    
    y_pred = intercept + slope * x_clipped
    residuals = y_clipped - y_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((y_clipped - y_mean) ** 2)
    r2 = 1 - (rss / tss) if tss > 0 else 0
    
    # Calculate standard error of the slope for confidence metric
    n = len(x_clipped)
    mse = rss / (n - 2) if n > 2 else 0
    std_error = np.sqrt(mse / Sxx) if Sxx > 0 and mse > 0 else 0
    
    return slope, intercept, r2, std_error

def get_latest_data_for_predictions(df):
    df['Report Date'] = pd.to_datetime(df['Report Date'])
    cutoff_date = datetime(2024, 7, 1)
    recent_df = df[df['Report Date'] >= cutoff_date].copy()
    
    if len(recent_df) == 0:
        print("Warning: No data found from Q2 2024 onwards. Using latest available data.")
        recent_df = df[df['Report Date'] >= df['Report Date'].max() - timedelta(days=365)].copy()
        cutoff_info = f"Latest year (from {df['Report Date'].max() - timedelta(days=365):%Y-%m-%d})"
    else:
        cutoff_info = "Q2 2024 onwards (from 2024-07-01)"
    
    latest_df = recent_df.loc[recent_df.groupby('Ticker')['Report Date'].idxmax()].copy()
    latest_df['Quarter'] = latest_df['Report Date'].dt.quarter
    latest_df['Year'] = latest_df['Report Date'].dt.year
    
    return latest_df, cutoff_info

def build_prediction_model(df, index_filter=None, use_fcf=False):
    if index_filter:
        df = df[df['Ticker'].isin(INDEX_TICKERS[index_filter])].copy()
    elif index_filter is None:  # Mega caps filter
        df = df[df['Market_Cap'] >= 200e9].copy()  # $200B+ market cap
    
    if use_fcf:
        x_column = '1Y_FCFps_growth'
        metric_name = "FCF"
    else:
        x_column = '1Y_Revenue_growth'
        metric_name = "Revenue"
    
    model_df = df[['Ticker', 'Market_Cap', x_column, '1Y_Price_growth']].dropna()
    
    if len(model_df) < 50:
        group_name = index_filter or 'Mega caps'
        print(f"Warning: Insufficient data for {group_name} model ({len(model_df)} observations)")
        return None, None, 0, 0, metric_name
    
    x_vals = model_df[x_column].values * 100
    y_vals = model_df['1Y_Price_growth'].values * 100
    
    slope, intercept, r2, std_error = calculate_regression_coefficients(x_vals, y_vals)
    
    return slope, intercept, r2, std_error, metric_name

def predict_stock_performance(latest_df, slope, intercept, index_filter=None, use_fcf=False, std_error=0):
    if slope is None or intercept is None:
        return pd.DataFrame()
    
    if index_filter:
        latest_df = latest_df[latest_df['Ticker'].isin(INDEX_TICKERS[index_filter])].copy()
    elif index_filter is None:  # Mega caps filter
        latest_df = latest_df[latest_df['Market_Cap'] >= 200e9].copy()
    
    if use_fcf:
        x_column = '1Y_FCFps_growth'
        growth_col_name = 'FCF_Growth_Pct'
    else:
        x_column = '1Y_Revenue_growth'
        growth_col_name = 'Revenue_Growth_Pct'
    
    pred_df = latest_df[['Ticker', 'Market_Cap', 'Quarter', 'Year', x_column]].dropna().copy()
    
    if len(pred_df) == 0:
        return pd.DataFrame()
    
    pred_df[growth_col_name] = pred_df[x_column] * 100
    pred_df['Predicted_Price_Change'] = intercept + slope * pred_df[growth_col_name]
    
    # Calculate t-statistic for statistical significance
    if std_error > 0:
        pred_df['t_statistic'] = abs(slope / std_error)  # Absolute t-statistic
    else:
        pred_df['t_statistic'] = 0.0  # No statistical significance
    
    pred_df = pred_df[pred_df['Predicted_Price_Change'] > 0].copy()
    pred_df = pred_df.sort_values('Predicted_Price_Change', ascending=False)
    
    return pred_df[['Ticker', 'Market_Cap', 'Quarter', 'Year', growth_col_name, 'Predicted_Price_Change', 't_statistic']]

def format_market_cap(market_cap):
    return f"${market_cap/1e9:.1f}B"

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Predict stock performance based on Revenue/FCF growth")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top predictions to show per group")
    parser.add_argument("--min-prediction", type=float, default=5.0, help="Minimum predicted price change percentage to include")
    parser.add_argument("--use-fcf", action="store_true", help="Use FCF growth instead of Revenue growth")
    args = parser.parse_args()

    metric_name = "FCF" if args.use_fcf else "Revenue"
    
    print("="*80)
    print("STOCK PRICE PREDICTION ANALYSIS")
    print(f"Based on {metric_name} Growth patterns from Q2 2024 to present")
    print("="*80)

    df = pd.read_csv(CSV, parse_dates=["Report Date"])
    latest_df, cutoff_info = get_latest_data_for_predictions(df)
    print(f"Using {len(latest_df)} companies with recent data for predictions")
    print(f"Data cutoff: {cutoff_info}")
    
    quarter_dist = latest_df.groupby(['Year', 'Quarter']).size().reset_index(name='Count')
    print("\nQuarter distribution of latest data used for predictions:")
    for _, row in quarter_dist.iterrows():
        print(f"  Q{row['Quarter']} {row['Year']}: {row['Count']} companies")
    print()

    groups = {
        "Mega caps": None,
        "S&P 500": "S&P 500", 
        "NASDAQ-100": "Nasdaq-100",
        "Dow Jones 30": "Dow Jones 30"
    }

    all_results = {}

    for group_name, index_filter in groups.items():
        print(f"\n{'='*60}")
        print(f"ANALYSIS: {group_name.upper()}")
        print(f"{'='*60}")
        
        slope, intercept, r2, std_error, metric_used = build_prediction_model(df, index_filter, args.use_fcf)
        
        if slope is None:
            print(f"Unable to build model for {group_name}")
            continue
            
        print(f"Model Statistics:")
        print(f"  Slope (β): {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Std Error: {std_error:.4f}")
        print(f"  Interpretation: Each 1% increase in {metric_used} growth predicts {slope:.2f}% price change")
        print()
        
        predictions = predict_stock_performance(latest_df, slope, intercept, index_filter, args.use_fcf, std_error)
        
        if len(predictions) == 0:
            print(f"No positive predictions found for {group_name}")
            continue
        
        # Filter by minimum prediction threshold
        predictions = predictions[predictions['Predicted_Price_Change'] >= args.min_prediction]
        
        growth_col = 'FCF_Growth_Pct' if args.use_fcf else 'Revenue_Growth_Pct'
        
        print(f"TOP {min(args.top_n, len(predictions))} PREDICTED WINNERS:")
        print(f"{'Rank':<4} {'Ticker':<8} {'Market Cap':<12} {'Quarter':<10} {f'{metric_used} Growth':<15} {'Predicted Gain':<15} {'t-stat':<8}")
        print("-" * 83)
        
        top_predictions = predictions.head(args.top_n)
        for i, (_, row) in enumerate(top_predictions.iterrows(), 1):
            quarter_info = f"Q{row['Quarter']} {row['Year']}"
            print(f"{i:<4} {row['Ticker']:<8} {format_market_cap(row['Market_Cap']):<12} "
                  f"{quarter_info:<10} {row[growth_col]:>8.1f}%{'':<6} {row['Predicted_Price_Change']:>10.1f}% {row['t_statistic']:>8.2f}")
        
        all_results[group_name] = {
            'model_stats': {'slope': slope, 'intercept': intercept, 'r2': r2, 'std_error': std_error, 'metric': metric_used},
            'predictions': top_predictions,
            'total_positive': len(predictions)
        }
        
        print(f"\nTotal companies with positive predictions: {len(predictions)}")

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Group':<15} {'Model R²':<10} {'Top Pick':<8} {'Top Gain':<10} {'Total Positive':<15}")
    print("-" * 80)
    
    for group_name, results in all_results.items():
        if results['predictions'].empty:
            continue
        top_pick = results['predictions'].iloc[0]
        print(f"{group_name:<15} {results['model_stats']['r2']:<10.3f} "
              f"{top_pick['Ticker']:<8} {top_pick['Predicted_Price_Change']:>7.1f}% "
              f"{results['total_positive']:<15}")

    # Overall top recommendations
    print(f"\n{'='*80}")
    print("OVERALL TOP RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Collect all predictions and sort
    all_predictions = []
    for group_name, results in all_results.items():
        if not results['predictions'].empty:
            group_predictions = results['predictions'].copy()
            group_predictions['Group'] = group_name
            all_predictions.append(group_predictions)
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        combined_predictions = combined_predictions.sort_values('Predicted_Price_Change', ascending=False)
        
        # Remove duplicates, keeping the highest predicted gain for each ticker
        combined_predictions = combined_predictions.drop_duplicates(subset='Ticker', keep='first')
        
        growth_col = 'FCF_Growth_Pct' if args.use_fcf else 'Revenue_Growth_Pct'
        
        print(f"TOP 10 OVERALL RECOMMENDATIONS:")
        print(f"{'Rank':<4} {'Ticker':<8} {'Group':<15} {'Quarter':<10} {f'{metric_name} Growth':<15} {'Predicted Gain':<15} {'t-stat':<8}")
        print("-" * 93)
        
        for i, (_, row) in enumerate(combined_predictions.head(10).iterrows(), 1):
            quarter_info = f"Q{row['Quarter']} {row['Year']}"
            print(f"{i:<4} {row['Ticker']:<8} {row['Group']:<15} "
                  f"{quarter_info:<10} {row[growth_col]:>8.1f}%{'':<6} {row['Predicted_Price_Change']:>10.1f}% {row['t_statistic']:>8.2f}")

if __name__ == "__main__":
    main()
