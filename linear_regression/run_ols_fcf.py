# run_ols_fcf.py
"""
OLS slope test: YoY_Price_growth  ~  beta0 + beta1 * YoY_FCFps_growth
Uses the formulas from the course sheet (no external stats library).
"""

import pandas as pd
import numpy as np
from scipy import stats   # only for the p-value of the t-stat

CSV = "../fcf_dataset.csv"   # adjust path if needed

# 1  load and clean ----------------------------------------------------------
df = pd.read_csv(CSV)

# keep only the two series we need and drop NaNs / inf
cols = ["YoY_Price_growth", "YoY_FCFps_growth"]
df = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
y = df["YoY_Price_growth"].values
x = df["YoY_FCFps_growth"].values
n = len(x)

# --- Winsorise both series to avoid wild outliers ---
p_low, p_high = 0.01, 0.99
x = np.clip(x, *np.quantile(x, [p_low, p_high]))
y = np.clip(y, *np.quantile(y, [p_low, p_high]))


# because of the low correlation, we also clip the data to a reasonable range
# we take giant outliers out.
mask = np.abs(x) <= 1
x, y = x[mask]*100, y[mask]          # A + B

# 2  compute slope and intercept (closed-form) -------------------------------
x_bar, y_bar = x.mean(), y.mean()
Sxx = np.sum((x - x_bar) ** 2)
Sxy = np.sum((x - x_bar) * (y - y_bar))

beta1_hat = Sxy / Sxx
beta0_hat = y_bar - beta1_hat * x_bar

# 3  standard error, t-stat, p-value ----------------------------------------
y_hat = beta0_hat + beta1_hat * x
resid = y - y_hat
s2 = np.sum(resid ** 2) / (n - 2)          # residual MSE
SE_beta1 = np.sqrt(s2 / Sxx)
t_stat = beta1_hat / SE_beta1
p_val = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

# 4  95 % CI for beta1 --------------------------------------------------------
t_crit = stats.t.ppf(0.975, df=n - 2)
ci_low  = beta1_hat - t_crit * SE_beta1
ci_high = beta1_hat + t_crit * SE_beta1

# 5  report ------------------------------------------------------------------
print("n observations        :", n)
print("beta1 (slope)         :", beta1_hat)
print(f"SE(beta1)             : {float(SE_beta1):.8f}")
print("t-statistic           :", t_stat)
print(f"p-value               : {float(p_val):.11f}")
print("95 % CI for beta1     : [{:.4g}, {:.4g}]".format(ci_low, ci_high))
print("beta0 (intercept)     :", beta0_hat)


# 6  Visualization -----------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Scatter plot of the clipped data
plt.scatter(x, y, alpha=0.3, edgecolors='none', label='Winsorized data')

# Regression line: use the closed-form beta0 + beta1
x_line = np.linspace(x.min(), x.max(), 100)
y_line = beta0_hat + beta1_hat * x_line
plt.plot(x_line, y_line, color='red', lw=2, label='OLS fit')

# Add titles and labels
plt.title(f"OLS Fit: YoY_Price_growth vs. YoY_FCFps_growth\n"
          f"Slope: {beta1_hat:.5f}  |  SE: {SE_beta1:.5f}  |  p = {p_val:.3f}")
plt.xlabel("YoY FCF per Share Growth (winsorized)")
plt.ylabel("YoY Price Growth (winsorized)")
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
# plt.hexbin(x, y, gridsize=60, cmap='Blues', mincnt=3)   # C

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
