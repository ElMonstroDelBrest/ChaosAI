#!/usr/bin/env python3
"""Download macro-economic data: FRED (rates, M2, CPI) + Yahoo (VIX, DXY, SPY).

All data saved as parquet in data/raw/macro/.
Light download — runs in seconds.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/raw/macro")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Yahoo Finance: VIX, DXY, S&P500 (daily + intraday) ──────────────────

YAHOO_TICKERS = {
    "VIX": "^VIX",           # CBOE Volatility Index
    "DXY": "DX-Y.NYB",       # US Dollar Index
    "SPY": "SPY",             # S&P 500 ETF (more liquid than ^GSPC)
    "SP500": "^GSPC",         # S&P 500 Index
    "GOLD": "GC=F",           # Gold Futures
    "US10Y": "^TNX",          # 10-Year Treasury Yield
    "BTC_USD": "BTC-USD",     # BTC reference price (Yahoo)
    "ETH_USD": "ETH-USD",     # ETH reference price (Yahoo)
}


def download_yahoo():
    """Download daily history for all tickers (max period)."""
    log.info("=== Yahoo Finance (daily, max history) ===")
    for name, ticker in YAHOO_TICKERS.items():
        out_file = OUTPUT_DIR / f"yahoo_{name}_daily.parquet"
        if out_file.exists():
            log.info(f"  {name}: already exists, skipping")
            continue
        try:
            df = yf.download(ticker, period="max", interval="1d", progress=False)
            if df.empty:
                log.warning(f"  {name}: no data returned")
                continue
            df.to_parquet(out_file, compression="zstd")
            log.info(f"  {name}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            log.warning(f"  {name}: error {e}")

    # Also get 1m intraday for VIX and DXY (last 7 days only — Yahoo limit)
    log.info("=== Yahoo Finance (1m intraday, last 7 days) ===")
    for name, ticker in [("VIX", "^VIX"), ("DXY", "DX-Y.NYB"), ("SPY", "SPY")]:
        out_file = OUTPUT_DIR / f"yahoo_{name}_1m.parquet"
        if out_file.exists():
            log.info(f"  {name} 1m: already exists, skipping")
            continue
        try:
            df = yf.download(ticker, period="7d", interval="1m", progress=False)
            if df.empty:
                log.warning(f"  {name} 1m: no data returned")
                continue
            df.to_parquet(out_file, compression="zstd")
            log.info(f"  {name} 1m: {len(df)} rows")
        except Exception as e:
            log.warning(f"  {name} 1m: error {e}")


# ── FRED: Macro rates (free, no API key needed for basic series) ─────────

FRED_SERIES = {
    "FED_RATE": "FEDFUNDS",       # Federal Funds Effective Rate
    "M2": "M2SL",                 # M2 Money Supply
    "CPI": "CPIAUCSL",            # Consumer Price Index
    "UNEMPLOYMENT": "UNRATE",     # Unemployment Rate
    "INFLATION_EXPECT": "T10YIE", # 10Y Breakeven Inflation
}


def download_fred():
    """Download FRED data via pandas-datareader or direct CSV."""
    log.info("=== FRED Macro Data ===")
    for name, series_id in FRED_SERIES.items():
        out_file = OUTPUT_DIR / f"fred_{name}.parquet"
        if out_file.exists():
            log.info(f"  {name}: already exists, skipping")
            continue
        try:
            # FRED provides free CSV access without API key
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url)
            date_col = [c for c in df.columns if "date" in c.lower()][0]
            df = df.rename(columns={date_col: "DATE"})
            df["DATE"] = pd.to_datetime(df["DATE"])
            df = df.set_index("DATE")
            df.columns = [name]
            # Drop missing values (FRED uses "." for missing)
            df = df[df[name] != "."]
            df[name] = df[name].astype(float)
            df.to_parquet(out_file, compression="zstd")
            log.info(f"  {name}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
        except Exception as e:
            log.warning(f"  {name}: error {e}")


if __name__ == "__main__":
    download_yahoo()
    download_fred()

    # Summary
    files = list(OUTPUT_DIR.glob("*.parquet"))
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    log.info(f"=== Macro data: {len(files)} files, {total_mb:.1f} MB in {OUTPUT_DIR} ===")
