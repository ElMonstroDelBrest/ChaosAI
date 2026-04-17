#!/usr/bin/env python3
"""Convert Kaggle CSV datasets to .pt tensors (same format as download_massive_data.py).

Each output file is a (T, 5) float32 tensor: [Open, High, Low, Close, Volume].
Sorted by timestamp, NaN rows dropped.

Supports:
  - US Stocks (jacksoncrow/stock-market-dataset): individual CSVs per ticker
  - Forex 1min: individual CSVs per pair
  - Generic OHLCV CSVs with standard column names

Usage on data-prep VM:
    python3 convert_kaggle_to_pt.py \
        --input_dir /tmp/data/stocks/ \
        --output_dir /tmp/data/stocks_pt/ \
        --source stocks

    python3 convert_kaggle_to_pt.py \
        --input_dir /tmp/data/forex/ \
        --output_dir /tmp/data/forex_pt/ \
        --source forex
"""

import argparse
import os
import glob
import sys

import numpy as np
import pandas as pd
import torch


def detect_ohlcv_columns(df):
    """Auto-detect OHLCV columns from various naming conventions."""
    col_map = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for target, candidates in {
        "open": ["open", "o", "price_open", "open_price"],
        "high": ["high", "h", "price_high", "high_price"],
        "low": ["low", "l", "price_low", "low_price"],
        "close": ["close", "c", "price_close", "close_price", "adj close", "adj_close"],
        "volume": ["volume", "v", "vol", "trade_volume"],
    }.items():
        for cand in candidates:
            if cand in cols_lower:
                col_map[target] = cols_lower[cand]
                break

    return col_map


def detect_timestamp_column(df):
    """Auto-detect timestamp column."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in ["timestamp", "date", "datetime", "time", "date_time", "dt"]:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def convert_csv_to_pt(csv_path, output_dir, min_rows=100):
    """Convert a single CSV to .pt tensor. Returns True if successful."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"  SKIP {csv_path}: read error: {e}")
        return False

    if len(df) < min_rows:
        return False

    # Detect columns
    col_map = detect_ohlcv_columns(df)
    if len(col_map) < 4:  # need at least OHLC
        print(f"  SKIP {csv_path}: only found columns {list(col_map.keys())}")
        return False

    # Sort by timestamp if available
    ts_col = detect_timestamp_column(df)
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)

    # Extract OHLCV
    ohlcv_cols = []
    for key in ["open", "high", "low", "close", "volume"]:
        if key in col_map:
            ohlcv_cols.append(col_map[key])
        elif key == "volume":
            # Volume optional — fill with zeros
            df["_volume_zero"] = 0.0
            ohlcv_cols.append("_volume_zero")
        else:
            return False

    data = df[ohlcv_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()

    if len(data) < min_rows:
        return False

    tensor = torch.tensor(data.values, dtype=torch.float32)

    # Output filename: use CSV stem
    name = os.path.splitext(os.path.basename(csv_path))[0]
    # Clean name (remove spaces, special chars)
    name = name.replace(" ", "_").replace("/", "_").upper()
    output_path = os.path.join(output_dir, f"{name}.pt")
    torch.save(tensor, output_path)
    return True


def convert_stocks(input_dir, output_dir):
    """Convert jacksoncrow/stock-market-dataset (individual CSVs per ticker)."""
    # Structure: stocks/etfs/*.csv and stocks/stocks/*.csv
    csv_files = []
    for subdir in ["etfs", "stocks", "ETFs", "Stocks", ""]:
        pattern = os.path.join(input_dir, subdir, "*.csv")
        csv_files.extend(glob.glob(pattern))

    if not csv_files:
        # Try flat directory
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    ok = 0
    for i, csv_path in enumerate(csv_files):
        if convert_csv_to_pt(csv_path, output_dir):
            ok += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(csv_files)} ({ok} converted)")

    print(f"Converted {ok}/{len(csv_files)} files to {output_dir}")
    return ok


def convert_forex(input_dir, output_dir):
    """Convert forex 1-minute CSVs."""
    csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    ok = 0
    for csv_path in csv_files:
        if convert_csv_to_pt(csv_path, output_dir, min_rows=1000):
            ok += 1

    print(f"Converted {ok}/{len(csv_files)} files to {output_dir}")
    return ok


def convert_generic(input_dir, output_dir):
    """Convert any directory of OHLCV CSVs."""
    csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    ok = 0
    for csv_path in csv_files:
        if convert_csv_to_pt(csv_path, output_dir):
            ok += 1

    print(f"Converted {ok}/{len(csv_files)} files to {output_dir}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Convert Kaggle CSV to .pt OHLCV tensors")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", choices=["stocks", "forex", "generic"], default="generic")
    parser.add_argument("--min_rows", type=int, default=100)
    args = parser.parse_args()

    converters = {
        "stocks": convert_stocks,
        "forex": convert_forex,
        "generic": convert_generic,
    }
    converters[args.source](args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
