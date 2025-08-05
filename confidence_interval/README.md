# Confidence Interval Analysis

This script performs confidence interval analysis for the relationship between growth metrics (FCF or Revenue) and forward price changes, segmented by market cap tiers.

## Usage

```bash
python ci.py [options]
```

## Options

- `--analysis-mode {fcf_growth,revenue_growth}`: Choose analysis type (default: revenue_growth)
- `--sp500`: Filter to S&P 500 stocks only
- `--nasdaq`: Filter to NASDAQ-100 stocks only  
- `--dow30`: Filter to Dow Jones 30 stocks only

## Examples

```bash
# Default: Revenue Growth analysis for all stocks
python ci.py

# FCF Growth analysis for S&P 500 stocks
python ci.py --analysis-mode fcf_growth --sp500

# Revenue Growth analysis for NASDAQ-100 stocks
python ci.py --analysis-mode revenue_growth --nasdaq

# FCF Growth analysis for Dow 30 stocks
python ci.py --analysis-mode fcf_growth --dow30
```

## Output

The script generates:
1. **Console output**: Slope coefficients and 95% confidence intervals for each market cap tier and time horizon
2. **Visual plot**: Confidence interval plot showing slopes and error bars by tier
3. **Summary table**: Statistical significance indicators (*** for CI not including zero)

## Market Cap Tiers

- **Micro-caps**: Bottom 10% by market cap
- **Mid-caps**: Middle 80% by market cap  
- **Mega-caps**: Top 10% by market cap
- **All Samples**: Complete dataset

## Time Horizons

- **1Y**: 1-year forward price growth vs 1-year growth metric
- **2Y**: 2-year forward price growth vs 2-year growth metric

## Statistical Notes

- Uses OLS regression with 95% confidence intervals
- Outliers trimmed at 3rd and 97th percentiles
- Requires minimum 10 observations per tier
- Statistical significance indicated when confidence interval excludes zero
