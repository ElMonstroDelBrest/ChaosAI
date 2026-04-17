"""Download on-chain transaction data from BigQuery public datasets.

Usage:
    python scripts/download_onchain_data.py --chain btc --start 2020-01-01 --end 2025-12-31
    python scripts/download_onchain_data.py --chain eth --start 2020-01-01 --end 2025-12-31
    python scripts/download_onchain_data.py --chain all  # both chains

Output: data/onchain/raw/{chain}/{YYYY-MM-DD_HH}.parquet

BigQuery public datasets are free up to 1 TB/month of query processing.
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


BTC_QUERY = """
SELECT * FROM (
    SELECT
        (SELECT addresses[SAFE_OFFSET(0)]
         FROM UNNEST(t.inputs)
         WHERE ARRAY_LENGTH(addresses) > 0
         LIMIT 1) AS from_address,
        outp.addresses[SAFE_OFFSET(0)] AS to_address,
        outp.output_satoshis / 1e8 AS value,
        t.hash AS tx_hash,
        TIMESTAMP_TRUNC(t.block_timestamp, HOUR) AS hour
    FROM `bigquery-public-data.crypto_bitcoin.transactions` AS t,
        UNNEST(t.outputs) AS outp
    WHERE t.block_timestamp >= @start_ts
      AND t.block_timestamp < @end_ts
      AND outp.output_satoshis > 0
      AND ARRAY_LENGTH(outp.addresses) > 0
)
WHERE from_address IS NOT NULL
  AND to_address IS NOT NULL
"""

ETH_QUERY = """
SELECT
    from_address,
    to_address,
    CAST(value AS FLOAT64) / 1e18 AS value,
    `hash` AS tx_hash,
    TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE block_timestamp >= @start_ts
  AND block_timestamp < @end_ts
  AND from_address IS NOT NULL
  AND to_address IS NOT NULL
  AND value > 0
"""


def download_chain(
    chain: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    max_bytes_billed: int = 1_000_000_000_000,
):
    """Download on-chain data for a single chain, day by day.

    Args:
        chain: "btc" or "eth".
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        output_dir: Base output directory.
        max_bytes_billed: BigQuery billing cap in bytes.
    """
    client = bigquery.Client()
    query_template = BTC_QUERY if chain == "btc" else ETH_QUERY

    out_path = Path(output_dir) / chain
    out_path.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    while current < end:
        next_day = current + timedelta(days=1)
        day_str = current.strftime("%Y-%m-%d")

        # Check if already downloaded
        existing = list(out_path.glob(f"{day_str}_*.parquet"))
        if existing:
            print(f"  [{chain}] {day_str}: {len(existing)} files exist, skipping")
            current = next_day
            continue

        print(f"  [{chain}] Querying {day_str}...")

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", current),
                bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", next_day),
            ],
            maximum_bytes_billed=max_bytes_billed,
        )

        try:
            df = client.query(query_template, job_config=job_config).to_dataframe()
        except Exception as e:
            print(f"  [{chain}] Error on {day_str}: {e}")
            current = next_day
            continue

        if df.empty:
            print(f"  [{chain}] {day_str}: no data")
            current = next_day
            continue

        # Group by hour and save
        df["hour"] = pd.to_datetime(df["hour"])
        for hour_ts, hour_df in df.groupby("hour"):
            hour_str = hour_ts.strftime("%Y-%m-%d_%H")
            parquet_path = out_path / f"{hour_str}.parquet"
            hour_df.drop(columns=["hour"]).to_parquet(parquet_path, index=False)

        print(f"  [{chain}] {day_str}: {len(df)} transactions -> {len(df['hour'].unique())} hourly files")
        current = next_day


def main():
    parser = argparse.ArgumentParser(description="Download on-chain data from BigQuery")
    parser.add_argument("--chain", default="all", choices=["btc", "eth", "all"])
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/onchain/raw/", help="Output directory")
    parser.add_argument("--max_bytes", type=int, default=1_000_000_000_000, help="BigQuery billing cap")
    args = parser.parse_args()

    chains = ["btc", "eth"] if args.chain == "all" else [args.chain]
    for chain in chains:
        print(f"\n=== Downloading {chain.upper()} transactions ===")
        download_chain(chain, args.start, args.end, args.output, args.max_bytes)

    print("\nDone.")


if __name__ == "__main__":
    main()
