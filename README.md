# Statistical Stock Market Test

This repository investigates whether **revenue growth can predict future stock price movements** for U.S. public companies.  
Using quarterly data from over 5,000 firms (2020–2024), we apply regression analysis and statistical tests across different market capitalizations and indices.

Our main finding: **revenue growth is a statistically significant predictor of future returns**, especially for large-cap and index-listed firms, though explanatory power remains modest.

---

## Key Results

### Revenue Growth vs. Price Changes Across Horizons
Revenue growth shows consistent positive predictive power, with stronger relationships in mega-cap firms and longer horizons.

<img src="written/images/all_horizons_all_stocks_single_plot.png" width="850">

### Index-Level Analysis (1Y Horizon)
<details>
<summary>Large, mature indices display stronger revenue–price links.</summary>

- **S&P 500**  
<img src="written/images/1_year_sp500_plot.png" width="400">

- **NASDAQ-100**  
<img src="written/images/1_year_nasdaq100_plot.png" width="400">

- **Dow Jones 30**  
<img src="written/images/1_year_dow30_plot.png" width="400">

### Confidence Intervals
95% confidence intervals confirm robustness of slopes, especially for large-cap firms at 1–2 year horizons.

<img src="written/images/all_horizons_confidence_intervals.png" width="600">
</details>

## Interactive site

A GitHub Pages dashboard ([almogtavor.github.io/statistical-stock-market-test](https://almogtavor.github.io/statistical-stock-market-test/)) maps the top 500 U.S. companies by forward P/E (NTM) vs 1-year revenue growth, and backtests the core thesis: ranking S&P 500 stocks by revenue growth and measuring the realized forward return of the top picks vs the S&P average. The top-3 basket beats the index by ~13pp at the 1Y horizon and ~23pp at 2Y.

<img src="top500_pe_ntm_vs_revenue_growth.png" width="850">

Build it with `python build_top_companies_chart.py` (static chart) and `python build_site_data.py` (site data).

## Resources

- 📄 `written/stats_paper.pdf`
- Processed dataset: [open-stock-reports-dataset](https://huggingface.co/datasets/almogtavor/open-stock-reports-dataset)
- Includes quarterly financials + stock prices for 5,000+ U.S. public companies (2020–2026).
