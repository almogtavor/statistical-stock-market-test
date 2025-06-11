#!/usr/bin/env python3
"""
Produce fcf_quarterly_merged.csv with:
Ticker, ReportDate, Price, FCF, FCF_per_share, FCFps_growth
"""

import os, time, pandas as pd, numpy as np, simfin as sf
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import HTTPError

# ---------- CONFIG ----------
MARKET, START_YEAR = 'US', 2000
DATA_DIR, OUT_CSV  = Path('simfin_data'), Path('fcf_quarterly_merged.csv')
MAX_RETRY, RETRY_DELAY = 5, 8
CAPEX_CANDS = [
    'Change in Fixed Assets & Intangibles',
    'Purchase of PPE & Intangibles, net',
    'Capital Expenditures (Fixed Assets)',
    'Capital Expenditures'
]
# -----------------------------

load_dotenv()
sf.set_api_key(os.getenv('SIMFIN_API_KEY', 'free'))
sf.set_data_dir(str(DATA_DIR))

def retry(func, *a, **kw):
    for i in range(MAX_RETRY):
        try: return func(*a, **kw)
        except HTTPError as e:
            if i == MAX_RETRY-1 or not 500 <= e.response.status_code < 600:
                raise
            wait = RETRY_DELAY * 2**i
            print(f'  ↻  {e.response.status_code}: retry in {wait}s …')
            time.sleep(wait)

# ---------- 1) fundamentals ----------
print('Cash-flow & income …')
cf  = retry(sf.load_cashflow, variant='quarterly', market=MARKET, refresh_days=365)
inc = retry(sf.load_income,   variant='quarterly', market=MARKET, refresh_days=365)
cf, inc = cf.reset_index(), inc.reset_index()

capex_col = next((c for c in CAPEX_CANDS if c in cf.columns), None)
if not capex_col:
    raise KeyError(f'CapEx column not found. Tried {CAPEX_CANDS}')

cf = cf[['Ticker', 'Report Date', 'Net Cash from Operating Activities', capex_col]]
cf.rename(columns={'Net Cash from Operating Activities': 'OCF', capex_col: 'CAPEX'},
          inplace=True)
cf['FCF'] = cf['OCF'] + cf['CAPEX']          # CAPEX ≤ 0
inc = inc[['Ticker', 'Report Date', 'Shares (Basic)']]

fund = (cf.merge(inc, on=['Ticker', 'Report Date'])
          .loc[lambda d: d['Report Date'].dt.year >= START_YEAR])
fund['FCF_per_share'] = fund['FCF'] / fund['Shares (Basic)']

fund.sort_values(['Ticker', 'Report Date'], inplace=True, ignore_index=True)
fund['FCFps_lag4']   = fund.groupby('Ticker')['FCF_per_share'].shift(4)
fund['FCFps_growth'] = (fund['FCF_per_share'] - fund['FCFps_lag4']) / fund['FCFps_lag4']

# ---------- 2) prices ----------
# ---------- 2) prices ----------
print('Daily prices …')
px = retry(sf.load_shareprices, variant='daily', market=MARKET, refresh_days=365)
px = (px.reset_index()[['Ticker','Date','Adj. Close']]
          .assign(Date=lambda d: pd.to_datetime(d['Date']))
          .sort_values(['Ticker','Date']))

# helper: map each (Ticker, ReportDate) to the first trade ≤ 7 days ahead
print('Aligning prices …')
# 1) restrict price table to a 7-day look-ahead window for speed
px['Date_plus7'] = px['Date'] - pd.Timedelta(days=7)

# 2) build a key for fast join
fund_keyed = (fund[['Ticker','Report Date']]
              .rename(columns={'Report Date':'RptDate'}))
px_keyed   = (px[['Ticker','Date','Adj. Close']]
              .rename(columns={'Date':'TradeDate','Adj. Close':'Price'}))

# 3) inner merge where TradeDate ≥ RptDate
temp = (fund_keyed.merge(px_keyed, on='Ticker')
                   .loc[lambda df: (df['TradeDate'] >= df['RptDate']) & (df['TradeDate'] <= (df['RptDate'] + pd.Timedelta("7D")))]
                   .sort_values(['Ticker','RptDate','TradeDate'])
                   .groupby(['Ticker','RptDate'], as_index=False)
                   .first())      # earliest valid price

merged = (fund.merge(temp, left_on=['Ticker','Report Date'],
                             right_on=['Ticker','RptDate'],
                             how='left')
                .drop(columns='RptDate'))

# (optional) forward 12-month price – identical trick but shift RptDate by +365 d


print('Merging & pruning …')
# merged = attach_prices(fund)
merged.dropna(subset=['Price', 'FCFps_growth'], inplace=True)

# Uncomment if you want these for later analysis
# merged['Pct_change_12m'] = (merged['Price_fwd'] - merged['Price']) / merged['Price']

final = merged[['Ticker', 'Report Date', 'Price',
                'FCF', 'FCF_per_share', 'FCFps_growth']]

print(f'Rows: {len(final):,}')
final.to_csv(OUT_CSV, index=False)
print(f'Saved → {OUT_CSV}\n', final.head())