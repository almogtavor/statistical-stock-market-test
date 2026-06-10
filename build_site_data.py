#!/usr/bin/env python3
"""
build_site_data.py

Generate docs/site_data.json for the GitHub Pages site:
  - scatter   : top-N-by-market-cap points (P/E NTM vs 1Y revenue growth)
  - logos     : base64 logos for the megacaps (Plotly layout images)
  - recs      : current top S&P 500 picks by 6M / 1Y / 2Y revenue growth
  - backtest  : aggregate edge of the top-3 strategy vs the S&P average,
                plus the most recent fully-measurable historical example.

Signal = trailing revenue growth from stocks_dataset.csv. Forward return =
realized price change over the matching horizon (6M=2q, 1Y=4q, 2Y=8q).
Revenue-growth artifacts (|g| > CAP, from near-zero/negative bases) are dropped.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import pandas as pd

import build_top_companies_chart as chart   # reuse universe selection + logos

CSV = "stocks_dataset.csv"
PE_CACHE = "forward_pe_cache.csv"
SP_LIST = "sp500_list.json"
OUT = "docs/site_data.json"

HORIZONS = {"6M": 2, "1Y": 4, "2Y": 8}
SIG = {h: f"{h}_Revenue_growth" for h in HORIZONS}
CAP = 8.0          # drop revenue-growth artifacts above 800%
N_CURRENT = 5      # current picks shown per signal
N_PICK = 3         # backtest basket size
MIN_CROSS = 20     # min S&P names reporting on a date to count it


def scatter_points() -> tuple[list, list]:
    """Reuse the chart's top-N universe; return point dicts + megacap logos."""
    df = pd.read_csv(CSV, parse_dates=["Report Date"])
    latest = chart.latest_rows(df)
    cand = latest.dropna(subset=["Market_Cap"])
    cand = cand[cand["Market_Cap"] <= chart.MCAP_CEILING].nlargest(chart.CANDIDATE_N, "Market_Cap")
    pe = pd.read_csv(PE_CACHE)
    m = cand.merge(pe, on="Ticker", how="left")
    m["mcap"] = pd.to_numeric(m["liveMarketCap"], errors="coerce")
    m = m[(m["mcap"] > 0) & (m["mcap"] <= chart.MCAP_CEILING)]
    m["fwd_pe"] = pd.to_numeric(m["forwardPE"], errors="coerce")
    m["rev_g"] = pd.to_numeric(m["1Y_Revenue_growth"], errors="coerce")
    uni = m.nlargest(chart.TOP_N, "mcap")
    plot = uni[(uni["fwd_pe"] > 0) & np.isfinite(uni["fwd_pe"]) & uni["rev_g"].notna()]

    pts = [{"t": r.Ticker,
            "pe": round(float(r.fwd_pe), 2),
            "g": round(float(r.rev_g), 4),
            "mc": round(float(r.mcap) / 1e9, 1)}
           for r in plot.itertuples()]

    logos = []
    for r in plot.nlargest(chart.N_LOGOS, "mcap").itertuples():
        path = os.path.join(chart.LOGO_DIR, f"{r.Ticker}.png")
        if not os.path.exists(path):
            chart.get_logo(r.Ticker)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            logos.append({"t": r.Ticker, "pe": round(float(r.fwd_pe), 2),
                          "g": round(float(r.rev_g), 4),
                          "img": f"data:image/png;base64,{b64}"})
    return pts, logos


def enrich_recs(tickers: list[str], ttm_rev: pd.Series) -> dict:
    """Fetch company name, current market cap and P/S for the recommended tickers.

    P/S uses Yahoo's trailing-12M figure when present, else market cap / TTM
    revenue from the dataset. Results cached to recs_enrich_cache.csv.
    """
    import yfinance as yf

    cache = {}
    if os.path.exists("recs_enrich_cache.csv"):
        c = pd.read_csv("recs_enrich_cache.csv").set_index("t")
        cache = {t: row.dropna().to_dict() for t, row in c.iterrows()}

    out = {}
    for t in tickers:
        if t in cache and cache[t].get("name"):
            out[t] = cache[t]
            continue
        name, mcap, ps = t, np.nan, np.nan
        try:
            info = yf.Ticker(t).get_info()
            name = info.get("longName") or info.get("shortName") or t
            mcap = info.get("marketCap", np.nan)
            ps = info.get("priceToSalesTrailing12Months", np.nan)
        except Exception as ex:
            print(f"  enrich {t}: {ex}")
        if (ps is None or pd.isna(ps)) and pd.notna(mcap) and ttm_rev.get(t, 0):
            ps = mcap / ttm_rev[t]
        rec = {"name": name,
               "mcap": (None if pd.isna(mcap) else round(float(mcap) / 1e9, 1)),
               "ps": (None if ps is None or pd.isna(ps) else round(float(ps), 1))}
        out[t] = rec
    pd.DataFrame([{"t": k, **v} for k, v in out.items()]).to_csv(
        "recs_enrich_cache.csv", index=False)
    return out


def recs_and_backtest():
    df = pd.read_csv(CSV, parse_dates=["Report Date"])
    sp = set(json.load(open(SP_LIST)))
    d = df[df["Ticker"].isin(sp)].copy().sort_values(["Ticker", "Report Date"])
    g = d.groupby("Ticker")
    for h, q in HORIZONS.items():
        d[f"fwd_{h}"] = g["Price"].shift(-q) / d["Price"] - 1

    # trailing-twelve-month revenue per ticker (last 4 quarters) for P/S fallback
    ttm_rev = g["Revenue"].apply(lambda s: s.tail(4).sum())

    # current picks (latest row per ticker)
    latest = d.groupby("Ticker").tail(1)
    current = {}
    for h, sig in SIG.items():
        ok = latest[latest[sig].notna() & (latest[sig].abs() <= CAP)]
        top = ok.nlargest(N_CURRENT, sig)
        current[h] = [{"t": r.Ticker, "g": round(float(r[sig]), 4)}
                      for _, r in top.iterrows()]

    # enrich every recommended ticker with name, current market cap and P/S
    enrich = enrich_recs(sorted({p["t"] for hs in current.values() for p in hs}), ttm_rev)
    for hs in current.values():
        for p in hs:
            p.update(enrich.get(p["t"], {}))

    # backtest: aggregate edge + most recent measurable example
    backtest = {}
    for h, sig in SIG.items():
        rows = d[d[sig].notna() & (d[sig].abs() <= CAP) & d[f"fwd_{h}"].notna()]
        per = []
        for T, snap in rows.groupby("Report Date"):
            if len(snap) < MIN_CROSS:
                continue
            top = snap.nlargest(N_PICK, sig)
            per.append((T, top[f"fwd_{h}"].mean(), snap[f"fwd_{h}"].mean(), top))
        if not per:
            continue
        arr = np.array([(a, b) for _, a, b, _ in per])
        last_T, _, last_bench, last_top = max(per, key=lambda x: x[0])
        example = {
            "date": last_T.strftime("%Y-%m-%d"),
            "bench": round(float(last_bench), 4),
            "picks": [{"t": r.Ticker, "sig": round(float(r[sig]), 4),
                       "ret": round(float(r[f"fwd_{h}"]), 4)}
                      for _, r in last_top.iterrows()],
        }
        backtest[h] = {
            "n": len(per),
            "top": round(float(arr[:, 0].mean()), 4),
            "bench": round(float(arr[:, 1].mean()), 4),
            "excess": round(float((arr[:, 0] - arr[:, 1]).mean()), 4),
            "winrate": round(float((arr[:, 0] > arr[:, 1]).mean()), 3),
            "example": example,
        }
    return current, backtest


def main() -> None:
    os.makedirs("docs", exist_ok=True)
    pts, logos = scatter_points()
    current, backtest = recs_and_backtest()
    data = {
        "asOf": pd.read_csv(CSV, usecols=["Report Date"])["Report Date"].max(),
        "topN": chart.TOP_N,
        "scatter": pts,
        "logos": logos,
        "recs": current,
        "backtest": backtest,
        "horizons": list(HORIZONS),
        "capPct": int(CAP * 100),
        "pickN": N_PICK,
    }
    with open(OUT, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    print(f"wrote {OUT}: {len(pts)} points, {len(logos)} logos, "
          f"backtest horizons {list(backtest)}")


if __name__ == "__main__":
    main()
