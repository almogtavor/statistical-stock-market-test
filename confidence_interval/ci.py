#!/usr/bin/env python3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

# Index ticker lists
INDEX_TICKERS = {
    "Nasdaq-100": ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS'],
    "Dow Jones 30": ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT'],
    "S&P 500": ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'XYZ', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'COIN', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DDOG', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TTD', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
}

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Confidence interval analysis for FCF and Revenue growth vs Price change")
    parser.add_argument("--analysis-mode", choices=["fcf_growth", "revenue_growth"], default="revenue_growth", 
                       help="Analysis mode: FCF growth or Revenue growth")
    parser.add_argument("--sp500", action="store_true", help="Filter to S&P 500 stocks only")
    parser.add_argument("--nasdaq", action="store_true", help="Filter to NASDAQ-100 stocks only") 
    parser.add_argument("--dow30", action="store_true", help="Filter to Dow Jones 30 stocks only")
    args = parser.parse_args()

    # Parameters
    P_HIGH = 97
    P_LOW = 3

    # Load data
    df = pd.read_csv('../fcf_dataset.csv')
    
    # Define analysis columns based on mode
    if args.analysis_mode == "revenue_growth":
        x_cols = ['1Y_Revenue_growth', '2Y_Revenue_growth']
        analysis_name = "Revenue Growth"
        x_label = "Revenue Growth (%)"
    else:  # fcf_growth
        x_cols = ['1Y_FCFps_growth', '2Y_FCFps_growth'] 
        analysis_name = "FCF Growth"
        x_label = "FCFps Growth (%)"
    
    # Select relevant columns
    base_cols = ['Ticker', 'Market_Cap', '1Y_Price_growth', '2Y_Price_growth']
    df = df[base_cols + x_cols]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Apply index filter if specified
    index_filter = None
    if args.sp500:
        df = df[df['Ticker'].isin(INDEX_TICKERS["S&P 500"])]
        index_filter = "S&P 500"
    elif args.nasdaq:
        df = df[df['Ticker'].isin(INDEX_TICKERS["Nasdaq-100"])]
        index_filter = "NASDAQ-100"
    elif args.dow30:
        df = df[df['Ticker'].isin(INDEX_TICKERS["Dow Jones 30"])]
        index_filter = "Dow 30"

    # Add forward-looking price columns
    df['1Y_Price_growth_lead'] = df.groupby('Ticker')['1Y_Price_growth'].shift(-1)
    df['2Y_Price_growth_lead'] = df.groupby('Ticker')['2Y_Price_growth'].shift(-2)
    df = df.dropna(subset=['1Y_Price_growth_lead', '2Y_Price_growth_lead'])

    print(f"Analysis: {analysis_name} vs Price Change")
    if index_filter:
        print(f"Index Filter: {index_filter}")
    print(f"Sample size after filtering: {len(df)} observations")

    # Define tiers
    top_10 = df['Market_Cap'].quantile(0.9)
    bottom_10 = df['Market_Cap'].quantile(0.1)

    micro_df = df[df['Market_Cap'] <= bottom_10].copy()
    mega_df = df[df['Market_Cap'] >= top_10].copy()
    mid_df = df[(df['Market_Cap'] > bottom_10) & (df['Market_Cap'] < top_10)].copy()

    # Add full data
    tiers = {
        "Micro": micro_df,
        "Mid": mid_df,
        "Mega": mega_df,
        "All": df
    }

    colors = {
        "Micro": "orange",
        "Mid": "green", 
        "Mega": "blue",
        "All": "black"
    }

    def collect_ci_slopes(df_dict, x_col, y_col, label_suffix):
        rows = []
        for name, subdf in df_dict.items():
            subdf = subdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
            if len(subdf) < 10:
                print(f"Skipping {name} {label_suffix}: insufficient data ({len(subdf)} observations)")
                continue
                
            x_low, x_high = subdf[x_col].quantile(P_LOW / 100), subdf[x_col].quantile(P_HIGH / 100)
            y_low, y_high = subdf[y_col].quantile(P_LOW / 100), subdf[y_col].quantile(P_HIGH / 100)
            subdf = subdf[
                (subdf[x_col] >= x_low) & (subdf[x_col] <= x_high) &
                (subdf[y_col] >= y_low) & (subdf[y_col] <= y_high)
            ]
            if len(subdf) < 10:
                print(f"Skipping {name} {label_suffix}: insufficient data after outlier removal ({len(subdf)} observations)")
                continue
                
            X = sm.add_constant(subdf[x_col])
            y = subdf[y_col]
            model = sm.OLS(y, X).fit()
            slope = model.params[x_col]
            ci = model.conf_int().loc[x_col].values
            print(f"{name} {label_suffix}: Slope = {slope:.4f}, CI = ({ci[0]:.4f}, {ci[1]:.4f}), N = {len(subdf)}")
            rows.append({
                "Group": f"{name} {label_suffix}",
                "Slope": slope,
                "CI_Low": ci[0],
                "CI_High": ci[1],
                "Color": colors[name],
                "N": len(subdf)
            })
        return pd.DataFrame(rows)

    # Collect confidence intervals for both horizons
    ci_df_1y = collect_ci_slopes(tiers, x_cols[0], '1Y_Price_growth_lead', '1Y')
    ci_df_2y = collect_ci_slopes(tiers, x_cols[1], '2Y_Price_growth_lead', '2Y')
    ci_df = pd.concat([ci_df_1y, ci_df_2y]).reset_index(drop=True)

    if len(ci_df) == 0:
        print("No valid data for confidence interval analysis")
        return

    # Plot
    plt.figure(figsize=(12, 8))
    for i, row in ci_df.iterrows():
        plt.plot([row['CI_Low'], row['CI_High']], [i, i], color='gray', linewidth=2)
        plt.scatter(row['Slope'], i, color=row['Color'], s=100, marker='o', 
                   edgecolors='black', linewidth=1, zorder=3)

    plt.yticks(range(len(ci_df)), ci_df['Group'])
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel(f"OLS Slope ({x_label} vs. Price Growth)", fontsize=12)
    
    # Create title with filter information
    title = f"Slope & Confidence Intervals by Market Cap Tier ({analysis_name})"
    if index_filter:
        title += f" - {index_filter} Only"
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Create legend
    legend_elements = [
        mpatches.Patch(color=colors['Micro'], label='Micro-caps'),
        mpatches.Patch(color=colors['Mid'], label='Mid-caps'),
        mpatches.Patch(color=colors['Mega'], label='Mega-caps'),
        mpatches.Patch(color=colors['All'], label='All Samples')
    ]
    plt.legend(handles=legend_elements, title="Market Cap Tier", loc="lower right")
    
    # Add sample size information
    plt.figtext(0.02, 0.02, f"Note: Error bars show 95% confidence intervals. Sample sizes vary by tier.", 
                fontsize=10, style='italic')
    
    # Save plot
    filename = f"confidence_intervals_{args.analysis_mode.lower()}"
    if index_filter:
        filename += f"_{index_filter.lower().replace(' ', '_').replace('&', 'and')}"
    filename += ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    plt.show()

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"CONFIDENCE INTERVAL SUMMARY - {analysis_name.upper()}")
    if index_filter:
        print(f"Index Filter: {index_filter}")
    print(f"{'='*60}")
    for _, row in ci_df.iterrows():
        significant = "***" if row['CI_Low'] > 0 or row['CI_High'] < 0 else ""
        print(f"{row['Group']:<15}: Slope = {row['Slope']:7.4f}, CI = [{row['CI_Low']:7.4f}, {row['CI_High']:7.4f}] {significant}")
    print(f"{'='*60}")
    print("*** indicates confidence interval does not include zero (statistically significant)")


if __name__ == "__main__":
    main()
