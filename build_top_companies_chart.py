#!/usr/bin/env python3
"""
build_top_companies_chart.py

Equivalent of the original 7-megacap "P/E Ratio (NTM) vs Revenue (1Y Growth)"
bubble chart, but for the top N U.S. companies by market cap (TOP_N below).

- Universe + market cap + 1Y revenue growth come from stocks_dataset.csv
  (latest quarterly row per ticker).
- P/E Ratio (NTM) = forward P/E, fetched live from Yahoo Finance (not in the
  dataset), cached to forward_pe_cache.csv so re-runs are cheap.
- Top ~15 megacaps get circular logos; the rest are plotted as dots sized by
  market cap.
"""
from __future__ import annotations

import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FuncFormatter
import requests

warnings.simplefilter("ignore")

CSV = "stocks_dataset.csv"
PE_CACHE = "forward_pe_cache.csv"
LOGO_DIR = "logos"
OUT = "top500_pe_ntm_vs_revenue_growth.png"

TOP_N = 500               # companies to chart (by market cap, with valid fwd P/E)
CANDIDATE_N = 800         # fetch fwd P/E for this many (some lack estimates / are junk)
MCAP_CEILING = 1.0e13     # drop implausible market caps (Yahoo share-count glitches)
N_LOGOS = 15              # how many of the largest get a logo
GROWTH_LABEL = 0.90       # label any company whose 1Y revenue growth exceeds this
PE_VIEW_MAX = 200.0       # x-axis clip
REV_VIEW_MIN, REV_VIEW_MAX = -0.5, 3.0   # y-axis clip (-50%..+300%)

# Domain map for logos of the most likely top megacaps (Clearbit logo API).
LOGO_DOMAINS = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "NVDA": "nvidia.com",
    "GOOGL": "google.com", "GOOG": "google.com", "AMZN": "amazon.com",
    "META": "meta.com", "TSLA": "tesla.com", "AVGO": "broadcom.com",
    "BRK.B": "berkshirehathaway.com", "BRK-B": "berkshirehathaway.com",
    "BRK-A": "berkshirehathaway.com", "MU": "micron.com", "XOM": "exxonmobil.com",
    "LLY": "lilly.com", "JPM": "jpmorganchase.com", "V": "visa.com",
    "WMT": "walmart.com", "MA": "mastercard.com", "ORCL": "oracle.com",
    "XOM": "exxonmobil.com", "UNH": "unitedhealthgroup.com", "COST": "costco.com",
    "NFLX": "netflix.com", "HD": "homedepot.com", "PG": "pg.com",
    "JNJ": "jnj.com", "ABBV": "abbvie.com", "BAC": "bankofamerica.com",
    "CRM": "salesforce.com", "KO": "coca-cola.com", "AMD": "amd.com",
    "CVX": "chevron.com", "WFC": "wellsfargo.com", "PLTR": "palantir.com",
    "TSM": "tsmc.com", "ASML": "asml.com", "ADBE": "adobe.com",
    "PEP": "pepsico.com", "MRK": "merck.com", "TMUS": "t-mobile.com",
    "BABA": "alibaba.com", "TM": "toyota.com", "NVO": "novonordisk.com",
}


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Latest quarterly row per ticker."""
    df = df.sort_values(["Ticker", "Report Date"])
    return df.groupby("Ticker", as_index=False).tail(1)


def fetch_forward_pe(tickers: list[str]) -> pd.DataFrame:
    """Fetch forward P/E (and live market cap) from Yahoo, with on-disk cache."""
    import yfinance as yf

    cache = {}
    if os.path.exists(PE_CACHE):
        c = pd.read_csv(PE_CACHE)
        cache = {r.Ticker: (r.forwardPE, r.liveMarketCap) for r in c.itertuples()}

    rows = []
    todo = [t for t in tickers if t not in cache]
    print(f"forward P/E: {len(cache)} cached, {len(todo)} to fetch")
    for i, t in enumerate(todo, 1):
        fpe, mc = np.nan, np.nan
        try:
            info = yf.Ticker(t).get_info()
            fpe = info.get("forwardPE", np.nan)
            mc = info.get("marketCap", np.nan)
        except Exception as ex:
            print(f"  {t}: {ex}")
        cache[t] = (fpe, mc)
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}] fetched")
            pd.DataFrame([{"Ticker": k, "forwardPE": v[0], "liveMarketCap": v[1]}
                          for k, v in cache.items()]).to_csv(PE_CACHE, index=False)
        time.sleep(0.2)

    out = pd.DataFrame([{"Ticker": k, "forwardPE": v[0], "liveMarketCap": v[1]}
                        for k, v in cache.items()])
    out.to_csv(PE_CACHE, index=False)
    return out


def get_logo(ticker: str):
    os.makedirs(LOGO_DIR, exist_ok=True)
    path = os.path.join(LOGO_DIR, f"{ticker}.png")
    if os.path.exists(path):
        try:
            return plt.imread(path)
        except Exception:
            return None
    domain = LOGO_DOMAINS.get(ticker)
    if not domain:
        return None
    for url in (f"https://logo.clearbit.com/{domain}?size=128",
                f"https://www.google.com/s2/favicons?domain={domain}&sz=128"):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and r.content:
                with open(path, "wb") as fh:
                    fh.write(r.content)
                return plt.imread(io.BytesIO(r.content))
        except Exception:
            continue
    return None


def circular(img):
    """Crop an RGBA image to a circle."""
    if img is None:
        return None
    img = np.array(img, dtype=float)
    if img.max() > 1.0:
        img = img / 255.0
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = np.dstack([img, img, img, np.ones((h, w))])
    if img.shape[2] == 3:
        img = np.dstack([img, np.ones((h, w))])
    yy, xx = np.ogrid[:h, :w]
    cy, cx, r = h / 2, w / 2, min(h, w) / 2
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[..., 3] = img[..., 3] * mask
    return img


def main() -> None:
    df = pd.read_csv(CSV, parse_dates=["Report Date"])
    latest = latest_rows(df)

    # candidate universe by dataset market cap (ceiling-filter the worst glitches
    # so fetch budget is not wasted on absurd entries)
    cand = latest.dropna(subset=["Market_Cap"])
    cand = cand[cand["Market_Cap"] <= MCAP_CEILING].nlargest(CANDIDATE_N, "Market_Cap")
    pe = fetch_forward_pe(cand["Ticker"].tolist())
    m = cand.merge(pe, on="Ticker", how="left")

    # rank by LIVE market cap only (Yahoo's dataset share counts are glitchy);
    # require a plausible live value so junk listings drop out
    m["mcap"] = pd.to_numeric(m["liveMarketCap"], errors="coerce")
    m = m[(m["mcap"] > 0) & (m["mcap"] <= MCAP_CEILING)]

    # valid points: positive finite forward P/E and a revenue growth value
    m["fwd_pe"] = pd.to_numeric(m["forwardPE"], errors="coerce")
    m["rev_g"] = pd.to_numeric(m["1Y_Revenue_growth"], errors="coerce")
    universe = m.nlargest(TOP_N, "mcap")            # the top-1000 universe by mcap
    valid = universe[(universe["fwd_pe"] > 0) & np.isfinite(universe["fwd_pe"])
                     & universe["rev_g"].notna()]
    plot_df = valid.reset_index(drop=True)
    print(f"top-{TOP_N} universe: {len(universe)}, plottable (valid fwd P/E + rev g): "
          f"{len(plot_df)}")

    # clip to view window
    plot_df["x"] = plot_df["fwd_pe"].clip(upper=PE_VIEW_MAX)
    plot_df["y"] = plot_df["rev_g"].clip(REV_VIEW_MIN, REV_VIEW_MAX)

    # bubble size scaled by market cap
    mc = plot_df["mcap"].astype(float)
    sizes = 25 + 600 * (mc / mc.max()) ** 0.5

    # ---- plot ----
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=110)
    bg = "#1b1b1f"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.scatter(plot_df["x"], plot_df["y"], s=sizes, c="#4fa3ff",
               alpha=0.55, edgecolors="none", zorder=2)

    # logos for the largest N
    top = plot_df.nlargest(N_LOGOS, "mcap")
    logo_tickers = set(top["Ticker"])
    for r in top.itertuples():
        img = circular(get_logo(r.Ticker))
        if img is not None:
            zoom = 0.20 * (128 / max(img.shape[0], img.shape[1]))
            ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (r.x, r.y),
                                frameon=False, zorder=5)
            ax.add_artist(ab)
        else:
            ax.annotate(r.Ticker, (r.x, r.y), color="white", fontsize=7,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="#333", ec="none"))

    # ticker labels for the top-growth names (>GROWTH_LABEL revenue growth)
    hot = plot_df[plot_df["rev_g"] > GROWTH_LABEL]
    print(f"top-growth (>{GROWTH_LABEL:.0%}) labelled: {len(hot)}")
    for r in hot.itertuples():
        # nudge the text off the marker; below it if a logo already sits there
        dy = -0.045 if r.Ticker in logo_tickers else 0.0
        ax.annotate(r.Ticker, (r.x, r.y), xytext=(0, 6 if dy == 0 else -10),
                    textcoords="offset points", color="#ffd24a", fontsize=7,
                    ha="center", va="bottom" if dy == 0 else "top", zorder=6,
                    fontweight="bold")

    ax.set_title(f"P/E Ratio (NTM) vs Revenue (1Y Growth)  -  Top {TOP_N} by Market Cap",
                 color="#e8e8e8", fontsize=15, pad=16)
    ax.set_xlabel("P/E Ratio (NTM)", color="#cfcfcf", fontsize=12, fontweight="bold")
    ax.set_ylabel("Revenue (1Y Growth)", color="#cfcfcf", fontsize=12, fontweight="bold")
    ax.set_xlim(0, PE_VIEW_MAX)
    ax.set_ylim(REV_VIEW_MIN, REV_VIEW_MAX)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.grid(True, color="#33343a", linewidth=0.7, alpha=0.6)
    for s in ax.spines.values():
        s.set_color("#33343a")
    ax.tick_params(colors="#bdbdbd")

    fig.text(0.99, 0.965, f"TOP {TOP_N} - VJN AI", color="#3ddc84",
             fontsize=8, ha="right", va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, facecolor=bg, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
