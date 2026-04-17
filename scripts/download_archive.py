#!/usr/bin/env python3
"""Bulk download Binance Futures klines from data.binance.vision archive.

No API rate limits — direct HTTP file downloads. ~100x faster than API.

URL pattern (USDT-M Futures, monthly):
  https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY}-{MM}.zip

Each zip contains a CSV with columns:
  Open_time, Open, High, Low, Close, Volume, Close_time,
  Quote_asset_volume, Number_of_trades, Taker_buy_base, Taker_buy_quote, Ignore

Usage:
    # Download all 432 pairs, 1m, 2020-2025:
    python scripts/download_archive.py --interval 1m

    # Specific pairs:
    python scripts/download_archive.py --interval 1m --pairs BTCUSDT ETHUSDT

    # Different interval:
    python scripts/download_archive.py --interval 1h

    # Custom date range:
    python scripts/download_archive.py --interval 1m --start 2022-01 --end 2025-12

    # Convert to parquet after download:
    python scripts/download_archive.py --interval 1m --to-parquet
"""

import argparse
import csv
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def get_pairs_from_ohlcv(ohlcv_dir: str = "data/ohlcv_v5/") -> list[str]:
    """Get pair list from existing ohlcv_v5 .pt files."""
    p = Path(ohlcv_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.glob("*.pt"))


def get_pairs_from_api() -> list[str]:
    """Fallback: get active USDT-M perpetual pairs from Binance API."""
    resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10)
    resp.raise_for_status()
    symbols = []
    for s in resp.json()["symbols"]:
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING":
            symbols.append(s["symbol"])
    return sorted(symbols)


def generate_months(start: str, end: str) -> list[tuple[int, int]]:
    """Generate (year, month) tuples from start to end (inclusive).

    Args:
        start: "YYYY-MM" format.
        end: "YYYY-MM" format.

    Returns:
        List of (year, month) tuples.
    """
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    months = []
    current = s
    while current <= e:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def download_one(symbol: str, interval: str, year: int, month: int, output_dir: Path) -> str | None:
    """Download a single monthly zip file.

    Returns:
        Path to saved zip file, or None if not available (404).
    """
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    out_path = output_dir / symbol / filename

    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)  # Already downloaded

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return None  # Pair didn't exist yet for this month
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        return str(out_path)
    except requests.RequestException:
        return None


def zip_to_dataframe(zip_path: str) -> pd.DataFrame | None:
    """Extract CSV from zip and convert to DataFrame."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as f:
                # Peek first line to detect header
                first_line = f.readline().decode().strip()
                f.seek(0)
                has_header = first_line.startswith("open_time")
                col_names = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore",
                ]
                df = pd.read_csv(
                    f,
                    header=0 if has_header else None,
                    names=None if has_header else col_names,
                    dtype={
                        "open": float, "high": float, "low": float,
                        "close": float, "volume": float,
                    },
                )
                if has_header:
                    df.columns = col_names
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                return df
    except (zipfile.BadZipFile, Exception):
        return None


def convert_pair_to_parquet(symbol: str, zip_dir: Path, parquet_dir: Path) -> int:
    """Convert all monthly zips for a pair into a single parquet file.

    Returns:
        Number of rows written.
    """
    symbol_dir = zip_dir / symbol
    if not symbol_dir.exists():
        return 0

    zips = sorted(symbol_dir.glob("*.zip"))
    if not zips:
        return 0

    dfs = []
    for zp in zips:
        df = zip_to_dataframe(str(zp))
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        return 0

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("open_time").drop_duplicates(subset=["open_time"])

    parquet_dir.mkdir(parents=True, exist_ok=True)
    out_path = parquet_dir / f"{symbol}.parquet"
    merged.to_parquet(out_path, index=False)
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Bulk download from Binance archive")
    parser.add_argument("--interval", default="1m", help="Kline interval (1m, 5m, 1h, etc.)")
    parser.add_argument("--pairs", nargs="+", default=None, help="Specific pairs (default: all from ohlcv_v5)")
    parser.add_argument("--start", default="2020-01", help="Start month (YYYY-MM)")
    parser.add_argument("--end", default="2025-12", help="End month (YYYY-MM)")
    parser.add_argument("--output", default="data/raw/", help="Output base directory")
    parser.add_argument("--workers", type=int, default=32, help="Parallel download threads")
    parser.add_argument("--to-parquet", action="store_true", help="Convert zips to parquet after download")
    parser.add_argument("--parquet-dir", default=None, help="Parquet output dir (default: data/raw/{interval}_parquet/)")
    args = parser.parse_args()

    # Get pairs
    if args.pairs:
        pairs = args.pairs
    else:
        pairs = get_pairs_from_ohlcv()
        if not pairs:
            print("No pairs in ohlcv_v5/, fetching from API...")
            pairs = get_pairs_from_api()
    print(f"Pairs: {len(pairs)}")

    # Generate month list
    months = generate_months(args.start, args.end)
    print(f"Months: {len(months)} ({args.start} -> {args.end})")
    print(f"Total jobs: {len(pairs) * len(months):,} files")
    print(f"Workers: {args.workers}")

    zip_dir = Path(args.output) / f"{args.interval}_archive"

    # Build job list
    jobs = []
    for pair in pairs:
        for year, month in months:
            jobs.append((pair, args.interval, year, month, zip_dir))

    # Download
    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, *job): job for job in jobs}
        with tqdm(total=len(futures), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    downloaded += 1
                else:
                    skipped += 1  # 404 = pair didn't exist yet
                pbar.update(1)
                pbar.set_postfix(ok=downloaded, skip=skipped)

    print(f"\nDownloaded: {downloaded:,} | Skipped (404): {skipped:,}")

    # Convert to parquet
    if args.to_parquet:
        parquet_dir = Path(args.parquet_dir) if args.parquet_dir else Path(args.output) / f"{args.interval}_parquet"
        print(f"\nConverting to parquet: {parquet_dir}")
        total_rows = 0
        for pair in tqdm(pairs, desc="Converting"):
            rows = convert_pair_to_parquet(pair, zip_dir, parquet_dir)
            total_rows += rows
        print(f"Total rows: {total_rows:,}")

    print("Done.")


if __name__ == "__main__":
    main()
