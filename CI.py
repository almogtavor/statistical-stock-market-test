#!/usr/bin/env python3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    # Parameters
    P_HIGH = 97
    P_LOW = 3

    # Load data
    df = pd.read_csv('cash-time-machine/fcf_dataset.csv')
    df = df[['Ticker', 'Market_Cap', 'YoY_FCFps_growth', 'YoY_Price_growth',
             'YoY2_FCFps_growth', 'YoY2_Price_growth']]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Add lead columns
    df['YoY_Price_growth_lead'] = df.groupby('Ticker')['YoY_Price_growth'].shift(-1)
    df['YoY2_Price_growth_lead'] = df.groupby('Ticker')['YoY2_Price_growth'].shift(-2)
    df = df.dropna(subset=['YoY_Price_growth_lead', 'YoY2_Price_growth_lead'])

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
        "Micro": "blue",
        "Mid": "green",
        "Mega": "red",
        "All": "black"
    }

    def collect_ci_slopes(df_dict, x_col, y_col, label_suffix):
        rows = []
        for name, subdf in df_dict.items():
            subdf = subdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
            x_low, x_high = subdf[x_col].quantile(P_LOW / 100), subdf[x_col].quantile(P_HIGH / 100)
            y_low, y_high = subdf[y_col].quantile(P_LOW / 100), subdf[y_col].quantile(P_HIGH / 100)
            subdf = subdf[
                (subdf[x_col] >= x_low) & (subdf[x_col] <= x_high) &
                (subdf[y_col] >= y_low) & (subdf[y_col] <= y_high)
            ]
            if len(subdf) < 10:
                continue
            X = sm.add_constant(subdf[x_col])
            y = subdf[y_col]
            model = sm.OLS(y, X).fit()
            slope = model.params[x_col]
            ci = model.conf_int().loc[x_col].values
            print(f"{name} {label_suffix}: Slope = {slope:.4f}, CI = ({ci[0]:.4f}, {ci[1]:.4f})")
            rows.append({
                "Group": f"{name} {label_suffix}",
                "Slope": slope,
                "CI_Low": ci[0],
                "CI_High": ci[1],
                "Color": colors[name]
            })
        return pd.DataFrame(rows)

    ci_df_1y = collect_ci_slopes(tiers, 'YoY_FCFps_growth', 'YoY_Price_growth_lead', '1Y')
    ci_df_2y = collect_ci_slopes(tiers, 'YoY2_FCFps_growth', 'YoY2_Price_growth_lead', '2Y')
    ci_df = pd.concat([ci_df_1y, ci_df_2y]).reset_index(drop=True)

    # Plot
    plt.figure(figsize=(10, 6))
    for i, row in ci_df.iterrows():
        plt.plot([row['CI_Low'], row['CI_High']], [i, i], color='gray', linewidth=1)
        plt.scatter(row['Slope'], i, color=row['Color'], s=80, marker='o', label=row['Group'])

    plt.yticks(range(len(ci_df)), ci_df['Group'])
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("OLS Slope (FCFps Growth vs. Price Growth)")
    plt.title("Slope & Confidence Intervals by Market Cap Tier (1Y & 2Y)")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    legend_elements = [
        mpatches.Patch(color=colors['Micro'], label='Micro'),
        mpatches.Patch(color=colors['Mid'], label='Mid'),
        mpatches.Patch(color=colors['Mega'], label='Mega'),
        mpatches.Patch(color=colors['All'], label='All')
    ]
    plt.legend(handles=legend_elements, title="Tier", loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
