#!/usr/bin/env python3
"""
Create fcf_quarterly_merged.csv with:
  • Free-Cash-Flow per share (quarterly)
  • Matching daily price at the report date
  • One row per report   (2000-2024, US market)

Usage:
  export SIMFIN_API_KEY='YOUR_KEY'  # or leave unset for the public key
  python build_fcf_dataset.py
"""

import os, time, pandas as pd, numpy as np, simfin as sf
from pathlib import Path
from dotenv import load_dotenv
import time
from requests.exceptions import HTTPError

# Load environment variables from .env file
load_dotenv()

# -------------------- CONFIG --------------------
MARKET       = 'US'
START_YEAR   = 2000
DATA_DIR     = Path('simfin_data')
OUT_CSV      = Path('fcf_quarterly_merged.csv')
MAX_RETRIES  = 5
RETRY_DELAY  = 8          # seconds (exponential back-off)
# ------------------------------------------------

sf.set_api_key(os.getenv('SIMFIN_API_KEY', 'free'))
sf.set_data_dir(str(DATA_DIR))

def retry(func, *args, **kwargs):
    "Retry helper with exponential back-off."
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            if attempt == MAX_RETRIES - 1 or not (500 <= e.response.status_code < 600):
                raise
            wait = RETRY_DELAY * 2**attempt
            print(f'  ↻  Server error {e.response.status_code}. Retry in {wait}s …')
            time.sleep(wait)

# ---------- 1) fundamentals ----------
print('↳ Downloading quarterly cash-flow statements …')
try:
    cf = retry(sf.load_cashflow, variant='quarterly', market=MARKET, refresh_days=365)
except Exception as e:
    # ----- Offline fallback: Kaggle bulk CSV -----
    kaggle_zip = Path('simfin_cashflow_quarterly.zip')   # put the file next to this script
    if kaggle_zip.exists():
        print('  ⚠️  API failed. Loading local Kaggle mirror …')
        cf = pd.read_csv(kaggle_zip, compression='zip', parse_dates=['Report Date'])
    else:
        raise RuntimeError('Could not obtain cash-flow data.') from e

print('↳ Downloading quarterly income statements (for share count) …')
inc = retry(sf.load_income, variant='quarterly', market=MARKET, refresh_days=365)

# Merge CF and INCOME to get FCF per share
print('↳ Computing Free-Cash-Flow per share …')
cols_cf   = ['Ticker', 'Report Date',
             'Net Cash from Operating Activities',   # cash in
             'Purchase of Property, Plant & Equipment']  # CapEx (usually negative)
cf  = cf.reset_index()[cols_cf]
cf.rename(columns={
    'Net Cash from Operating Activities': 'OCF',
    'Purchase of Property, Plant & Equipment': 'CAPEX'
}, inplace=True)
cf['FCF'] = cf['OCF'] + cf['CAPEX']            # CAPEX is negative cash-out
inc = inc.reset_index()[['Ticker', 'Report Date', 'Shares (Basic)']]

fund = pd.merge(cf, inc, on=['Ticker', 'Report Date'], how='inner', validate='one_to_one')
fund['FCF_per_share'] = fund['FCF'] / fund['Shares (Basic)']
fund = fund[fund['Report Date'].dt.year >= START_YEAR]

# ---------- 2) prices ----------
print('↳ Downloading daily prices …')
px = retry(sf.load_shareprices, variant='daily', market=MARKET, refresh_days=365)
px = px.reset_index()[['Ticker', 'Date', 'Adj. Close']]

# ---------- 3) align price with report ----------
print('↳ Aligning price to report date …')
px = px.set_index(['Ticker', 'Date']).sort_index()
fund = fund.sort_values(['Ticker', 'Report Date'])
merged = pd.merge_asof(
    fund,
    px.reset_index().sort_values('Date'),
    by='Ticker',
    left_on='Report Date',
    right_on='Date',
    direction='forward',
    tolerance=pd.Timedelta('7D')
).rename(columns={'Adj. Close': 'Price'}).dropna(subset=['Price'])

print(f'✓ Final dataset: {merged.shape[0]:,} rows')
merged.to_csv(OUT_CSV, index=False)
print(f'✓ Saved → {OUT_CSV}')
print(merged.head())
