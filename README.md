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

## Resources

- 📄 `written/stats_paper.pdf`
- Processed dataset: [open-stock-reports-dataset](https://huggingface.co/datasets/almogtavor/open-stock-reports-dataset)
- Includes quarterly financials + stock prices for 5,000+ U.S. public companies (2020–2025).
