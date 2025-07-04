#!/usr/bin/env python3
"""
build_fcf_dataset_yfinance.py
-----------------------------

Create fcf_price_table.csv with columns:
Ticker, Report Date, Price, YoY_Price_growth, Market_Cap,
FCF, FCF_per_share, YoY_FCFps_growth

Compared with the SimFin version, this script:
  • Uses only free Yahoo Finance data via yfinance (no API key, no rate-limit headaches).
  • Automatically pulls a broad US-large-cap universe (S&P500 + NASDAQ100 + Dow30).
  • Goes back as far as Yahoo has quarterly cash-flow data (often well before 2015).
"""

from __future__ import annotations
from datetime import datetime, timedelta
import time
import sys
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from typing import List, Optional

###############################################################################
# ----------------------------- CONFIG -------------------------------------- #
###############################################################################
OUTFILE              = "fcf_price_table.csv"
PRICE_LOOKAHEAD_DAYS = 7          # how many trading days after report-date to accept
SLEEP_BETWEEN_TICKER = 0.3        # polite pause to avoid throttling
MAX_YFINANCE_RETRY   = 3

pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore", category=FutureWarning)

###############################################################################
# ---------------------  1.  Get a big ticker list  -------------------------- #
###############################################################################
def _scrape_table(url: str) -> List[str]:
    """Scrape the first HTML table from `url` and extract the first column that looks like tickers."""
    tables = pd.read_html(url)
    if not tables:
        return []
    df = tables[0]
    for col in ["Symbol", "Ticker"]:
        if col in df.columns:
            return list(df[col].astype(str).str.upper())
    # fallback: first column
    return list(df.iloc[:, 0].astype(str).str.upper())

def get_universe() -> List[str]:
    """Fetch union of S&P500, NASDAQ-100, and Dow 30 symbols."""
    print("Fetching ticker lists from Wikipedia ...")
    sp500  = _scrape_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    nasdaq = _scrape_table("https://en.wikipedia.org/wiki/NASDAQ-100")
    dow30  = _scrape_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    tickers = sorted(set(sp500) | set(nasdaq) | set(dow30))
    print(f"✅ Tickers found: S&P500={len(sp500)}, NASDAQ100={len(nasdaq)}, Dow30={len(dow30)}. Union={len(tickers)}")
    return tickers


###############################################################################
# -------- 2.  Robust helpers to pull Yahoo data with retries  --------------- #
###############################################################################
def yfin_retry(method, *args, **kwargs):
    for i in range(1, MAX_YFINANCE_RETRY + 1):
        try:
            out = method(*args, **kwargs)
            return out
        except Exception as e:
            if i == MAX_YFINANCE_RETRY:
                raise
            wait = i * 2
            print(f"    retry {i}/{MAX_YFINANCE_RETRY} after error: {e}  (sleep {wait}s)")
            time.sleep(wait)

def get_quarterly_cashflow(tkr: str) -> Optional[pd.DataFrame]:
    """
    Return DF with index=report date, columns=['FCF', 'Shares'].
    Tries 'Free Cash Flow'; if missing, computes OCF + CapEx.
    """
    yft = yf.Ticker(tkr)

    # ✅ Do NOT call yfin_retry here — this is a property, not a method
    cf = yft.quarterly_cashflow.T
    if cf.empty:
        return None

    if "Free Cash Flow" in cf.columns:
        cf["FCF"] = cf["Free Cash Flow"]
    elif {"Operating Cash Flow", "Capital Expenditures"}.issubset(cf.columns):
        cf["FCF"] = cf["Operating Cash Flow"] + cf["Capital Expenditures"]
    else:
        return None

    shares = yft.get_info().get("sharesOutstanding")
    if not shares:
        return None
    cf["Shares"] = int(shares)

    cf = cf[["FCF", "Shares"]].dropna()
    cf.index.name = "Report Date"
    return cf

def nearest_price(yft: yf.Ticker, rpt_dates: pd.Index) -> pd.Series:
    """
    For each report date, return the first close price on or after that date
    within PRICE_LOOKAHEAD_DAYS trading days.
    """
    start = (rpt_dates.min() - timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
    hist = yfin_retry(yft.history, start=start, end=end, interval="1d")

    if isinstance(hist, tuple):
        hist = hist[0]

    if hist.empty:
        return pd.Series(index=rpt_dates, dtype="float64")

    hist = hist[["Close"]]
    hist.index = pd.to_datetime(hist.index).tz_localize(None)

    close_for = []
    for d in rpt_dates:
        mask = (hist.index >= d) & (hist.index <= d + timedelta(days=PRICE_LOOKAHEAD_DAYS))
        sub = hist.loc[mask]
        close_for.append(sub["Close"].iloc[0] if not sub.empty else np.nan)
    return pd.Series(close_for, index=rpt_dates)


###############################################################################
# ---------------------- 3.  Per-ticker builder  ----------------------------- #
###############################################################################
def build_one_ticker(tkr: str) -> Optional[pd.DataFrame]:
    """
    Return DataFrame with required 8 columns for `tkr`, or None on failure/insufficient data.
    """
    cf = get_quarterly_cashflow(tkr)
    if cf is None or cf.empty:
        print(f"⚠️  {tkr:5}: no usable cash-flow data")
        return None

    yft = yf.Ticker(tkr)
    prices = nearest_price(yft, cf.index)

    df = cf.copy()
    df["Price"]       = prices
    df.dropna(subset=["Price"], inplace=True)

    if df.empty:
        print(f"⚠️  {tkr:5}: no matching prices")
        return None

    df["FCF_per_share"] = df["FCF"] / df["Shares"]
    df["Market_Cap"]    = df["Price"] * df["Shares"]

    df.sort_index(inplace=True)
    df["YoY_Price_growth"]  = df["Price"].pct_change(4)
    df["YoY_FCFps_growth"]  = df["FCF_per_share"].pct_change(4)

    df.reset_index(inplace=True)
    df.insert(0, "Ticker", tkr)

    out = df[["Ticker", "Report Date", "Price", "YoY_Price_growth",
              "Market_Cap", "FCF", "FCF_per_share", "YoY_FCFps_growth"]]
    return out

###############################################################################
# ---------------------------- 4.  Main run  --------------------------------- #
###############################################################################
def main(tickers: List[str]):
    frames = []
    for i, tkr in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tkr}")
        try:
            df = build_one_ticker(tkr)
            if df is not None and not df.empty:
                frames.append(df)
        except KeyboardInterrupt:
            print("Interrupted by user; exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  {tkr:5}: {e}")
        time.sleep(SLEEP_BETWEEN_TICKER)

    if not frames:
        print("No data retrieved - aborting.")
        return

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(OUTFILE, index=False, float_format="%.6f")
    print(f"\n✅ Done. {len(panel):,} rows written → {OUTFILE}")

###############################################################################
# --------------------------------------------------------------------------- #
###############################################################################
if __name__ == "__main__":
    universe = get_universe()
    print(f"Ticker universe size: {len(universe)}")
    main(universe)
