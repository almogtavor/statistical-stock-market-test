#!/usr/bin/env python3
"""
Produce fcf_quarterly_merged.csv with:
Ticker, Report Date, Price, 1Y_Price_growth, 2Y_Price_growth,
3Y_Price_growth, 6M_Price_growth,
Market_Cap, EV, Volume,
Revenue, Net Income,
FCF, FCF_per_share,
Yo6M_FCFps_growth, 1Y_FCFps_growth, 2Y_FCFps_growth, 3Y_FCFps_growth,
6M_NetIncome_growth, 1Y_NetIncome_growth, 2Y_NetIncome_growth, 3Y_NetIncome_growth,
6M_Volume_growth, 1Y_Volume_growth, 2Y_Volume_growth, 3Y_Volume_growth,
6M_Revenue_growth, 1Y_Revenue_growth, 2Y_Revenue_growth, 3Y_Revenue_growth
"""

import os, time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import simfin as sf
from dotenv import load_dotenv
from requests.exceptions import HTTPError

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
MARKET, START_YEAR = "US", 2000
DATA_DIR, OUT_CSV  = Path("simfin_data"), Path("fcf_dataset.csv")
MAX_RETRY, RETRY_DELAY = 5, 8
CAPEX_CANDS = [
    "Change in Fixed Assets & Intangibles",
    "Purchase of PPE & Intangibles, net",
    "Capital Expenditures (Fixed Assets)",
    "Capital Expenditures"
]

def retry(func, *args, **kwargs):
    for i in range(MAX_RETRY):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            if i == MAX_RETRY - 1 or not (500 <= e.response.status_code < 600):
                raise
            time.sleep(RETRY_DELAY * 2**i)

# ──────────────────────────────────────────────────────────────────────────────
# 1) FUNDAMENTALS
# ──────────────────────────────────────────────────────────────────────────────
print("Loading cash-flow, income & balance …")
sf.set_api_key(os.getenv("SIMFIN_API_KEY", "free"))
sf.set_data_dir(str(DATA_DIR))

cf  = retry(sf.load_cashflow,  variant="quarterly", market=MARKET, refresh_days=365)
inc = retry(sf.load_income,    variant="quarterly", market=MARKET, refresh_days=365)
bs  = retry(sf.load_balance,   variant="quarterly", market=MARKET, refresh_days=365)

cf, inc, bs = cf.reset_index(), inc.reset_index(), bs.reset_index()

capex_col = next((c for c in CAPEX_CANDS if c in cf.columns), None)
if not capex_col:
    raise KeyError(f"CapEx column not found. Tried {CAPEX_CANDS}")

cf = cf[["Ticker","Report Date","Net Cash from Operating Activities",capex_col]].rename(
    columns={"Net Cash from Operating Activities":"OCF", capex_col:"CAPEX"}
)
cf["FCF"] = cf["OCF"] + cf["CAPEX"]

inc = inc[["Ticker","Report Date","Shares (Basic)","Revenue","Net Income"]]

cash_col = "Cash, Cash Equivalents & Short Term Investments"
std_col  = "Short Term Debt"
ltd_col  = "Long Term Debt"

bs["Total Debt"] = bs[std_col].fillna(0) + bs[ltd_col].fillna(0)
bs = bs[["Ticker","Report Date","Total Debt",cash_col]]\
       .rename(columns={cash_col:"Cash and Cash Equivalents"})

fund = (
    cf.merge(inc, on=["Ticker","Report Date"])
      .merge(bs, on=["Ticker","Report Date"])
      .loc[lambda d: d["Report Date"].dt.year>=START_YEAR]
)
fund["FCF_per_share"] = fund["FCF"] / fund["Shares (Basic)"]

fund.sort_values(["Ticker","Report Date"], inplace=True, ignore_index=True)
grp = fund.groupby("Ticker")
for lag,label in [(2,"6M"),(4,"1Y"),(8,"2Y"),(12,"3Y")]:
    fund[f"FCFps_lag{lag}"] = grp["FCF_per_share"].shift(lag)
    fund[f"{label}_FCFps_growth"] = (
        fund["FCF_per_share"] - fund[f"FCFps_lag{lag}"]
    ) / fund[f"FCFps_lag{lag}"]

# ──────────────────────────────────────────────────────────────────────────────
# 2) PRICES
# ──────────────────────────────────────────────────────────────────────────────
print("Loading daily share prices …")
px = retry(sf.load_shareprices, variant="daily", market=MARKET, refresh_days=365)
px = (
    px.reset_index()[["Ticker","Date","Adj. Close","Volume"]]
      .rename(columns={"Date":"TradeDate","Adj. Close":"Price"})
      .assign(TradeDate=lambda d: pd.to_datetime(d["TradeDate"]))
      .sort_values(["Ticker","TradeDate"])
)

fund_keyed = fund[["Ticker","Report Date"]].rename(columns={"Report Date":"RptDate"})
print("Aligning prices within +-7 days")
temp = (
    fund_keyed.merge(px, on="Ticker", how="left")
      .loc[lambda d: (d["TradeDate"]>=d["RptDate"]) &
                     (d["TradeDate"]<=d["RptDate"]+pd.Timedelta(days=7))]
      .sort_values(["Ticker","RptDate","TradeDate"])
      .groupby(["Ticker","RptDate"], as_index=False).first()
)

merged = (
    fund.merge(temp, left_on=["Ticker","Report Date"],
                    right_on=["Ticker","RptDate"], how="left")
         .drop(columns="RptDate")
)
merged.dropna(subset=["Price"], inplace=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3) METRICS & SAVE
# ──────────────────────────────────────────────────────────────────────────────
print("Computing market cap, EV & growth metrics …")
merged["Market_Cap"] = merged["Price"] * merged["Shares (Basic)"]
merged["EV"]         = merged["Market_Cap"] + merged["Total Debt"] - merged["Cash and Cash Equivalents"]

grp2 = merged.groupby("Ticker")

# Price growth
for lag,label in [(2,"6M"),(4,"1Y"),(8,"2Y"),(12,"3Y")]:
    merged[f"Price_lag{lag}"] = grp2["Price"].shift(lag)
    merged[f"{label}_Price_growth"] = (
        merged["Price"] - merged[f"Price_lag{lag}"]
    ) / merged[f"Price_lag{lag}"]

# Net Income growth
for lag,label in [(2,"6M"),(4,"1Y"),(8,"2Y"),(12,"3Y")]:
    merged[f"NI_lag{lag}"] = grp2["Net Income"].shift(lag)
    merged[f"{label}_NetIncome_growth"] = (
        merged["Net Income"] - merged[f"NI_lag{lag}"]
    ) / merged[f"NI_lag{lag}"]

# Volume growth
for lag,label in [(2,"6M"),(4,"1Y"),(8,"2Y"),(12,"3Y")]:
    merged[f"Vol_lag{lag}"] = grp2["Volume"].shift(lag)
    merged[f"{label}_Volume_growth"] = (
        merged["Volume"] - merged[f"Vol_lag{lag}"]
    ) / merged[f"Vol_lag{lag}"]

# Revenue growth
for lag,label in [(2,"6M"),(4,"1Y"),(8,"2Y"),(12,"3Y")]:
    merged[f"Rev_lag{lag}"] = grp2["Revenue"].shift(lag)
    merged[f"{label}_Revenue_growth"] = (
        merged["Revenue"] - merged[f"Rev_lag{lag}"]
    ) / merged[f"Rev_lag{lag}"]

# final column order
final = merged[[
    "Ticker","Report Date",
    "Price","Volume","Market_Cap","EV",
    "6M_Price_growth","1Y_Price_growth","2Y_Price_growth","3Y_Price_growth",
    "Revenue","Net Income",
    "FCF","FCF_per_share",
    "6M_FCFps_growth","1Y_FCFps_growth","2Y_FCFps_growth","3Y_FCFps_growth",
    "6M_NetIncome_growth","1Y_NetIncome_growth","2Y_NetIncome_growth","3Y_NetIncome_growth",
    "6M_Volume_growth","1Y_Volume_growth","2Y_Volume_growth","3Y_Volume_growth",
    "6M_Revenue_growth","1Y_Revenue_growth","2Y_Revenue_growth","3Y_Revenue_growth"
]]

print(f"Rows: {len(final):,}")
final.to_csv(OUT_CSV, index=False, float_format="%.3f")
print(f"Saved → {OUT_CSV}")
