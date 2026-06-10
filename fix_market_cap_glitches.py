#!/usr/bin/env python3
"""
fix_market_cap_glitches.py

Repair isolated share-count glitches in stocks_dataset.csv.

Some rows carry a sharesOutstanding value that is ~1000x too large (a Yahoo
data glitch), which inflates Market_Cap, EV and FCF_per_share for that single
quarter while every other row of the same ticker is correct (e.g. BLKB
2023-03-31 shows a fake $3.68T cap, HCA 2024-03-31 a fake $86T cap).

Detection is purely internal: implied shares = Market_Cap / Price. A row is a
glitch when its implied share count exceeds GLITCH_FACTOR x the ticker's median
implied shares. Glitched rows get a corrected share count interpolated from the
ticker's good rows, and Market_Cap / EV / FCF_per_share are recomputed. Net debt
(EV - Market_Cap) is preserved, since both terms are inflated by the same factor.
Growth columns that depend on FCF_per_share are then recomputed.

Run with --dry-run to report without writing.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from build_stocks_dataset_2025_data import recompute_growth_for_ticker

CSV = "stocks_dataset.csv"
# Implied shares > this x the ticker median => unit glitch. Set high (50x) so we
# only touch physically-impossible one-quarter share jumps (the ~1000x Yahoo
# unit errors) and leave ambiguous real dilution/splits/mergers untouched.
GLITCH_FACTOR = 50.0


def fix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["Ticker", "Report Date"]).reset_index(drop=True)
    impl = df["Market_Cap"] / df["Price"]
    med = impl.groupby(df["Ticker"]).transform("median")
    glitch = (impl > GLITCH_FACTOR * med) & med.notna() & (med > 0)

    report = df.loc[glitch, ["Ticker", "Report Date", "Price", "Market_Cap"]].copy()
    report["old_impl_shares"] = impl[glitch]
    report["median_shares"] = med[glitch]

    if not glitch.any():
        return df, report

    # corrected share count: drop glitch rows, interpolate the rest per ticker
    good = impl.where(~glitch)
    corrected = good.groupby(df["Ticker"]).transform(
        lambda s: s.interpolate(limit_direction="both")
    )

    # only correct rows where we recovered a usable share count (>0, finite);
    # degenerate penny-stock tickers with no good rows are left as-is
    idx = df.index[glitch & (corrected > 0) & np.isfinite(corrected)]
    net_debt = df.loc[idx, "EV"] - df.loc[idx, "Market_Cap"]   # survives the inflation
    new_mcap = df.loc[idx, "Price"] * corrected[idx]
    df.loc[idx, "Market_Cap"] = new_mcap
    df.loc[idx, "EV"] = new_mcap + net_debt
    df.loc[idx, "FCF_per_share"] = df.loc[idx, "FCF"] / corrected[idx]
    report = report.loc[report.index.intersection(idx)]
    report["new_market_cap"] = new_mcap

    # recompute growth columns for affected tickers (FCFps growth depends on the fix)
    for tkr in report["Ticker"].unique():
        m = df["Ticker"] == tkr
        rec = recompute_growth_for_ticker(df.loc[m])
        df.loc[m, rec.columns] = rec.values

    return df, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["Report Date"])
    fixed, report = fix(df)

    print(f"glitched rows detected: {len(report)} across "
          f"{report['Ticker'].nunique() if len(report) else 0} tickers")
    if len(report):
        r = report.copy()
        r["factor"] = (r["old_impl_shares"] / r["median_shares"]).round(0)
        with pd.option_context("display.float_format", lambda v: f"{v:,.3g}"):
            print(r[["Ticker", "Report Date", "Market_Cap", "new_market_cap",
                     "factor"]].to_string(index=False))
    after = (fixed["Market_Cap"] / fixed["Price"])
    print(f"\nmax Market_Cap after fix: {fixed['Market_Cap'].max():,.0f}")

    if args.dry_run:
        print("\n[dry-run] not writing.")
        return
    fixed = fixed.sort_values(["Ticker", "Report Date"])
    fixed.to_csv(args.csv, index=False, float_format="%.3f")
    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
