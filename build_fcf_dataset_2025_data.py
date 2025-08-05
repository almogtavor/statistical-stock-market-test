#!/usr/bin/env python3
"""
build_fcf_dataset_2025_data.py

Incrementally extend fcf_dataset.csv with fresh Yahoo Finance data
and keep a resumable pointer in yahoo_fill_progress.txt.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

###############################################################################
# -------------------------------- CONFIG ----------------------------------- #
###############################################################################
MASTER_CSV = "fcf_dataset.csv"
PROGRESS_FILE = "yahoo_fill_progress.txt"
PRICE_LOOKAHEAD_DAYS = 7
SLEEP_BETWEEN_TICKER = 0.3
MAX_YFIN_RETRY       = 3

pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore", category=FutureWarning)

###############################################################################
# ----------------------------- I/O HELPERS --------------------------------- #
###############################################################################
def safe_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False, float_format="%.3f")
    os.replace(tmp, path)

def read_master() -> pd.DataFrame:
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"{MASTER_CSV} not found - create it first.")
    return pd.read_csv(MASTER_CSV, parse_dates=["Report Date"])

def save_progress(ticker: str) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
        fh.write(ticker + "\n")

def load_progress() -> str | None:
    if not os.path.exists(PROGRESS_FILE):
        return None
    with open(PROGRESS_FILE, "r", encoding="utf-8") as fh:
        last = fh.readline().strip()
        return last or None

###############################################################################
# -------------------- YAHOO FINANCE WRAPPER (RETRY) ------------------------ #
###############################################################################
def yfin_retry(method, *args, **kwargs):
    for i in range(1, MAX_YFIN_RETRY + 1):
        try:
            return method(*args, **kwargs)
        except Exception as ex:
            if i == MAX_YFIN_RETRY:
                raise
            wait = i * 2
            print(f"    retry {i}/{MAX_YFIN_RETRY} after: {ex} (sleep {wait}s)")
            time.sleep(wait)

def get_quarterly_cashflow(tkr: str) -> Optional[pd.DataFrame]:
    yft = yf.Ticker(tkr)
    cf = yft.quarterly_cashflow.T
    if cf.empty:
        return None

    if "Free Cash Flow" in cf.columns:
        cf["FCF"] = cf["Free Cash Flow"]
    elif {"Operating Cash Flow", "Capital Expenditures"} <= set(cf.columns):
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

def get_quarterly_income(tkr: str) -> Optional[pd.DataFrame]:
    inc = yf.Ticker(tkr).quarterly_income_stmt.T
    if inc.empty:
        return None
    # harmonise column names
    rev_col = next((c for c in ["Total Revenue", "Revenue"] if c in inc.columns), None)
    ni_col  = "Net Income" if "Net Income" in inc.columns else None
    if not rev_col or not ni_col:
        return None
    inc = inc[[rev_col, ni_col]].rename(columns={rev_col: "Revenue", ni_col: "Net Income"})
    inc.index.name = "Report Date"
    return inc

def nearest_price(yft: yf.Ticker, rpt_dates: pd.Index) -> pd.Series:
    start = (rpt_dates.min() - timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    hist  = yfin_retry(yft.history, start=start, end=end, interval="1d")
    if isinstance(hist, tuple):
        hist = hist[0]
    if hist.empty:
        return pd.Series(index=rpt_dates, dtype="float64")

    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    close_for = []
    for d in rpt_dates:
        mask = (hist.index >= d) & (hist.index <= d + timedelta(days=PRICE_LOOKAHEAD_DAYS))
        sub  = hist.loc[mask]
        close_for.append(sub["Close"].iloc[0] if not sub.empty else np.nan)
    return pd.Series(close_for, index=rpt_dates)

###############################################################################
# ---------------------- COMPUTE A SINGLE TICKER FRAME ---------------------- #
###############################################################################
def build_one_ticker(tkr: str) -> Optional[pd.DataFrame]:
    cf  = get_quarterly_cashflow(tkr)
    inc = get_quarterly_income(tkr)
    if cf is None or cf.empty or inc is None or inc.empty:
        print(f"⚠️  {tkr:5}: missing cash-flow or income data")
        return None

    df = cf.join(inc, how="inner")
    if df.empty:
        print(f"⚠️  {tkr:5}: no overlapping CF/INC dates")
        return None

    yft     = yf.Ticker(tkr)
    prices  = nearest_price(yft, df.index)
    df["Price"] = prices
    df.dropna(subset=["Price"], inplace=True)
    if df.empty:
        print(f"⚠️  {tkr:5}: no matching prices")
        return None

    df["FCF_per_share"] = df["FCF"] / df["Shares"]
    df["Market_Cap"]    = df["Price"] * df["Shares"]
    df.sort_index(inplace=True)

    df.reset_index(inplace=True)
    df.insert(0, "Ticker", tkr)

    cols = ["Ticker", "Report Date", "Price", "Market_Cap",
            "Revenue", "Net Income",
            "FCF", "FCF_per_share"]
    return df[cols]

###############################################################################
# ------------- RECOMPUTE GROWTH COLUMNS FOR A SINGLE TICKER --------------- #
###############################################################################
def recompute_growth_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """df – rows of one ticker only, sorted by Report Date."""
    df = df.sort_values("Report Date").copy()

    # price growth
    df["6M_Price_growth"] = df["Price"].pct_change(2)
    df["1Y_Price_growth"]  = df["Price"].pct_change(4)
    df["2Y_Price_growth"] = df["Price"].pct_change(8)
    df["3Y_Price_growth"] = df["Price"].pct_change(12)

    # FCF/share growth
    df["Yo6M_FCFps_growth"] = df["FCF_per_share"].pct_change(2)
    df["1Y_FCFps_growth"]  = df["FCF_per_share"].pct_change(4)
    df["2Y_FCFps_growth"] = df["FCF_per_share"].pct_change(8)
    df["3Y_FCFps_growth"] = df["FCF_per_share"].pct_change(12)

    return df

###############################################################################
# -------------------------------- MAIN ------------------------------------- #
###############################################################################
def main() -> None:
    master = read_master()
    tickers = sorted(master["Ticker"].unique())

    last_done = load_progress()
    if last_done:
        print(f"⏭️  Resuming after '{last_done}'")
        try:
            tickers = tickers[tickers.index(last_done) + 1:]
        except ValueError:
            pass

    if not tickers:
        print("Nothing to do - every ticker already processed.")
        return
    print(f"Tickers left: {len(tickers)}")

    for i, tkr in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tkr}")
        try:
            new_df = build_one_ticker(tkr)
            if new_df is None:
                save_progress(tkr)
                continue

            # Existing rows for this ticker
            current = master[master["Ticker"] == tkr]
            last_date = current["Report Date"].max() if not current.empty else None

            # Keep only truly new quarters
            mask = (new_df["Report Date"] > last_date) if pd.notna(last_date) else slice(None)
            delta = new_df.loc[mask]

            if delta.empty:
                print("   already up-to-date")
            else:
                master = pd.concat([master, delta], ignore_index=True)

            # Recompute growth metrics for this ticker (even if delta empty)
            idx = master["Ticker"] == tkr
            recomputed = recompute_growth_for_ticker(master.loc[idx])
            master.loc[idx, recomputed.columns] = recomputed

            # Sort and persist
            master.sort_values(["Ticker", "Report Date"], inplace=True)
            safe_write_csv(master, MASTER_CSV)
            print(f"   data rows: {len(recomputed)} (ticker total)")

            save_progress(tkr)
        except KeyboardInterrupt:
            print("Interrupted by user - progress saved.")
            sys.exit(0)
        except Exception as ex:
            print(f"⚠️  {tkr:5}: {ex}")
        time.sleep(SLEEP_BETWEEN_TICKER)

    print("\n✅ All done!")

if __name__ == "__main__":
    main()
