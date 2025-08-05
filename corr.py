
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load CSV
file_path = 'fcf_dataset.csv'  # Update path if needed
df = pd.read_csv(file_path)

x_val = 'Net Income'
y_val = 'Market_Cap'

# The given tickers list
# tickers = ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT']
tickers = ['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO', 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TMUS', 'TTWO', 'TSLA', 'TXN', 'TRI', 'TTD', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS']

# Filter for given tickers
df_filtered = df[df['Ticker'].isin(tickers)]

# Drop rows with NaN values for relevant columns
df_filtered = df_filtered.dropna(subset=[y_val, x_val])

# Use mean value for each ticker
df_filtered = df_filtered.groupby("Ticker")[[x_val, y_val]].mean().reset_index()

# Use yearly values
# df_filtered['Year'] = pd.to_datetime(df['Report Date']).dt.year
# df_yearly = df_filtered.groupby(['Ticker', 'Year'])[[x_val, y_val]].mean().reset_index()

# Prepare regression data
X = df_filtered[[x_val]].values.reshape(-1, 1)
y = df_filtered[y_val].values.reshape(-1, 1)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

correlation = np.corrcoef(df_filtered[x_val], df_filtered[y_val])[0, 1]
print(f"Correlation coefficient (r): {correlation:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('net income')
plt.ylabel('market cap')
plt.title('Linear Regression: Market Cap by Net Income')
plt.legend()
plt.grid(True)
plt.show()
