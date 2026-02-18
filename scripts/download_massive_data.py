#!/usr/bin/env python3
"""Massive Data Ingestion: download all Binance USDT-M Futures klines.

Auto-discovers all active USDT perpetual pairs, downloads complete history
(back to pair inception), detects/fills gaps, saves as .parquet per pair.

Features:
    - Async parallel downloads (aiohttp + semaphore for rate limiting)
    - Exponential backoff on errors / 429 / 418
    - Checkpointing: resumes from last downloaded timestamp per pair
    - Gap detection + forward-fill
    - Progress bars (tqdm) per pair and global

Usage:
    # Download all 1h data (default):
    PYTHONPATH=. python scripts/download_massive_data.py

    # Download 5m data:
    PYTHONPATH=. python scripts/download_massive_data.py --interval 5m

    # Limit to specific pairs (for testing):
    PYTHONPATH=. python scripts/download_massive_data.py --pairs BTCUSDT ETHUSDT

    # Dry run (list pairs only):
    PYTHONPATH=. python scripts/download_massive_data.py --dry_run

    # Resume interrupted download:
    PYTHONPATH=. python scripts/download_massive_data.py  # auto-resumes
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"

# Binance returns max 1500 klines per request
MAX_KLINES_PER_REQUEST = 1500

# Rate limiting: Binance Futures allows 2400 weight/min.
# Each klines request = weight 5 → max ~480 req/min.
# Be conservative: 10 concurrent requests, 150ms min between bursts.
MAX_CONCURRENT = 10
LIMIT_WEIGHT = 2400          # Binance 1-minute weight budget
PROACTIVE_THRESHOLD = 0.85   # Throttle proactively at 85% of weight budget
MIN_HISTORY_DAYS = 180  # 6 months minimum

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]

OHLCV_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
OHLCV_DTYPES = {
    "open": np.float64,
    "high": np.float64,
    "low": np.float64,
    "close": np.float64,
    "volume": np.float64,
}

INTERVAL_MS = {
    "1h": 3_600_000,
    "5m": 300_000,
    "15m": 900_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> list | dict | None:
    """Fetch JSON from Binance with retry + exponential backoff."""
    for attempt in range(max_retries):
        async with semaphore:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        # Proactive AIMD throttle: read Binance token-bucket header.
                        # All workers share the same IP weight budget, so any worker
                        # that sees >85% triggers a jittered pause to desynchronise
                        # the "Thundering Herd" before hitting the hard 429 wall.
                        used_weight = int(resp.headers.get("x-mbx-used-weight-1m", 0))
                        if used_weight > LIMIT_WEIGHT * PROACTIVE_THRESHOLD:
                            wait = 5.0 + random.uniform(0.0, 2.0)
                            log.info(
                                f"Proactive throttle: {used_weight}/{LIMIT_WEIGHT} weight used. "
                                f"Pausing {wait:.1f}s"
                            )
                            await asyncio.sleep(wait)
                        return await resp.json()
                    elif resp.status in (429, 418):
                        # Hard rate-limit hit — use Retry-After header + random jitter
                        # to desynchronise workers (avoid Thundering Herd on wake-up).
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        jitter = random.uniform(0.1, 0.3) * retry_after
                        wait_time = retry_after + jitter
                        log.warning(
                            f"Rate limited ({resp.status}). "
                            f"Worker pausing {wait_time:.1f}s (retry-after={retry_after}s + jitter)"
                        )
                        await asyncio.sleep(wait_time)
                    elif resp.status == 451:
                        # IP banned region
                        log.error(f"IP banned (451) for {params}")
                        return None
                    else:
                        log.warning(f"HTTP {resp.status} for {params}, retry {attempt+1}")
                        await asyncio.sleep(2 ** attempt)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning(f"Network error: {e}, retry {attempt+1}/{max_retries}")
                await asyncio.sleep(2 ** attempt)

    log.error(f"Failed after {max_retries} retries: {params}")
    return None


async def get_all_usdt_perpetuals(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> list[dict]:
    """Get all active USDT-M perpetual futures symbols."""
    data = await fetch_json(session, f"{BASE_URL}{EXCHANGE_INFO_ENDPOINT}", {}, semaphore)
    if not data:
        raise RuntimeError("Failed to fetch exchange info")

    pairs = []
    now_ms = int(time.time() * 1000)

    for s in data["symbols"]:
        if (
            s["contractType"] == "PERPETUAL"
            and s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
        ):
            onset_ms = s.get("onboardDate", now_ms)
            age_days = (now_ms - onset_ms) / 86_400_000
            pairs.append({
                "symbol": s["symbol"],
                "onboard_date": onset_ms,
                "age_days": age_days,
            })

    return pairs


# ---------------------------------------------------------------------------
# Download logic per pair
# ---------------------------------------------------------------------------

async def download_pair_klines(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pbar: tqdm | None = None,
) -> list[list]:
    """Download all klines for a single pair, paginating through history."""
    interval_ms = INTERVAL_MS[interval]
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": MAX_KLINES_PER_REQUEST,
        }

        data = await fetch_json(
            session, f"{BASE_URL}{KLINES_ENDPOINT}", params, semaphore,
        )

        if not data or len(data) == 0:
            break

        all_klines.extend(data)

        # Move start to after last received candle
        last_open_time = data[-1][0]
        current_start = last_open_time + interval_ms

        if pbar:
            pbar.update(len(data))

        # Delay between pagination requests to avoid 429s
        await asyncio.sleep(0.5)

        # If we got fewer than limit, we're done
        if len(data) < MAX_KLINES_PER_REQUEST:
            break

    return all_klines


def klines_to_dataframe(klines: list[list]) -> pd.DataFrame:
    """Convert raw klines to a clean DataFrame."""
    if not klines:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(klines, columns=KLINE_COLUMNS)

    # Keep only OHLCV columns
    df = df[OHLCV_COLUMNS].copy()

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates by timestamp
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    return df


def detect_and_fill_gaps(df: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, int]:
    """Detect temporal gaps and forward-fill them.

    Returns:
        (filled_df, n_gaps_filled)
    """
    if df.empty or len(df) < 2:
        return df, 0

    interval_td = pd.Timedelta(milliseconds=INTERVAL_MS[interval])

    # Create complete time index
    full_index = pd.date_range(
        start=df["open_time"].iloc[0],
        end=df["open_time"].iloc[-1],
        freq=interval_td,
    )

    n_expected = len(full_index)
    n_actual = len(df)
    n_gaps = n_expected - n_actual

    if n_gaps > 0:
        df = df.set_index("open_time")
        df = df.reindex(full_index)
        df.index.name = "open_time"

        # Forward-fill (causal — no future leakage)
        df = df.ffill()

        # If very first rows are NaN (gap at start), backfill just those
        df = df.bfill()

        df = df.reset_index()

    return df, n_gaps


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame as parquet with compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def load_checkpoint(path: Path) -> int | None:
    """Load existing parquet and return the last open_time as ms timestamp."""
    if not path.exists():
        return None
    try:
        table = pq.read_table(path, columns=["open_time"])
        df = table.to_pandas()
        if df.empty:
            return None
        last_ts = df["open_time"].max()
        if pd.isna(last_ts):
            return None
        return int(last_ts.timestamp() * 1000)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def download_one_pair(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    symbol: str,
    interval: str,
    output_dir: Path,
    onboard_ms: int,
    end_ms: int,
    global_pbar: tqdm,
) -> dict:
    """Download, clean, and save one pair. Returns stats dict."""
    out_path = output_dir / f"{symbol}.parquet"
    interval_ms = INTERVAL_MS[interval]

    # Checkpoint: resume from last downloaded timestamp
    checkpoint_ms = load_checkpoint(out_path)
    existing_df = None

    if checkpoint_ms is not None:
        start_ms = checkpoint_ms + interval_ms  # next candle after last
        if start_ms >= end_ms:
            global_pbar.update(1)
            return {"symbol": symbol, "status": "up-to-date", "candles": 0, "gaps": 0}
        # Load existing data to merge later
        try:
            existing_df = pq.read_table(out_path).to_pandas()
        except Exception:
            existing_df = None
            start_ms = onboard_ms
    else:
        start_ms = onboard_ms

    # Estimate total candles for this pair's sub-progress bar
    total_est = (end_ms - start_ms) // interval_ms

    try:
        klines = await download_pair_klines(
            session, semaphore, symbol, interval,
            start_ms=start_ms, end_ms=end_ms,
        )
    except Exception as e:
        log.error(f"{symbol}: download failed: {e}")
        global_pbar.update(1)
        return {"symbol": symbol, "status": f"error: {e}", "candles": 0, "gaps": 0}

    new_df = klines_to_dataframe(klines)

    # Merge with existing data if resuming
    if existing_df is not None and not new_df.empty:
        # Ensure same dtypes
        existing_df["open_time"] = pd.to_datetime(existing_df["open_time"], utc=True)
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    elif existing_df is not None:
        df = existing_df
    else:
        df = new_df

    if df.empty:
        global_pbar.update(1)
        return {"symbol": symbol, "status": "no_data", "candles": 0, "gaps": 0}

    # Gap detection + fill
    df, n_gaps = detect_and_fill_gaps(df, interval)

    # Save
    save_parquet(df, out_path)

    n_candles = len(df)
    first_date = df["open_time"].iloc[0]
    last_date = df["open_time"].iloc[-1]

    global_pbar.update(1)

    return {
        "symbol": symbol,
        "status": "ok",
        "candles": n_candles,
        "gaps": n_gaps,
        "first": str(first_date)[:10],
        "last": str(last_date)[:10],
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
    }


async def run_download(args):
    """Main async download orchestrator."""
    output_dir = Path(args.output_dir) / args.interval
    output_dir.mkdir(parents=True, exist_ok=True)

    max_conc = getattr(args, "max_concurrent", MAX_CONCURRENT)
    semaphore = asyncio.Semaphore(max_conc)

    connector = aiohttp.TCPConnector(limit=max_conc * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Discover pairs
        log.info("Fetching all USDT-M perpetual futures...")
        pairs = await get_all_usdt_perpetuals(session, semaphore)

        # Filter by minimum history
        pairs = [p for p in pairs if p["age_days"] >= MIN_HISTORY_DAYS]
        pairs.sort(key=lambda p: p["symbol"])

        # Filter by user-specified pairs
        if args.pairs:
            requested = set(args.pairs)
            pairs = [p for p in pairs if p["symbol"] in requested]

        log.info(f"Found {len(pairs)} pairs with >{MIN_HISTORY_DAYS}d history")

        if args.dry_run:
            print(f"\n{'Symbol':<15} {'Age (days)':<12} {'Onboard Date'}")
            print("-" * 50)
            for p in pairs:
                onboard = datetime.fromtimestamp(p["onboard_date"] / 1000, tz=timezone.utc)
                print(f"{p['symbol']:<15} {p['age_days']:<12.0f} {onboard.date()}")
            print(f"\nTotal: {len(pairs)} pairs")
            return

        # 2. Download all pairs in parallel
        end_ms = int(time.time() * 1000)

        log.info(f"Downloading {args.interval} klines for {len(pairs)} pairs...")
        log.info(f"Output: {output_dir}/")
        log.info(f"Max concurrent: {max_conc}")

        global_pbar = tqdm(total=len(pairs), desc="Pairs", unit="pair", position=0)

        tasks = []
        for p in pairs:
            task = download_one_pair(
                session, semaphore,
                symbol=p["symbol"],
                interval=args.interval,
                output_dir=output_dir,
                onboard_ms=p["onboard_date"],
                end_ms=end_ms,
                global_pbar=global_pbar,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        global_pbar.close()

    # 3. Summary
    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE — {args.interval} klines")
    print("=" * 70)

    ok = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    uptodate = [r for r in results if isinstance(r, dict) and r.get("status") == "up-to-date"]
    errors = [r for r in results if isinstance(r, dict) and r.get("status", "").startswith("error")]
    exceptions = [r for r in results if isinstance(r, Exception)]

    total_candles = sum(r["candles"] for r in ok)
    total_gaps = sum(r["gaps"] for r in ok)
    total_size = sum(r.get("size_mb", 0) for r in ok)

    print(f"\n  Downloaded:    {len(ok)} pairs ({total_candles:,} candles, {total_size:.1f} MB)")
    print(f"  Up-to-date:    {len(uptodate)} pairs (already complete)")
    print(f"  Gaps filled:   {total_gaps:,} candles (forward-fill)")
    print(f"  Errors:        {len(errors) + len(exceptions)} pairs")

    if ok:
        print(f"\n  {'Symbol':<15} {'Candles':>10} {'Gaps':>8} {'From':>12} {'To':>12} {'Size MB':>8}")
        print("  " + "-" * 67)
        for r in sorted(ok, key=lambda x: x["candles"], reverse=True)[:20]:
            print(f"  {r['symbol']:<15} {r['candles']:>10,} {r['gaps']:>8,} {r['first']:>12} {r['last']:>12} {r['size_mb']:>8}")
        if len(ok) > 20:
            print(f"  ... and {len(ok) - 20} more pairs")

    if errors:
        print(f"\n  Errors:")
        for r in errors:
            print(f"    {r['symbol']}: {r['status']}")

    if exceptions:
        print(f"\n  Exceptions:")
        for e in exceptions:
            print(f"    {e}")

    # Save manifest
    manifest_path = output_dir / "_manifest.csv"
    manifest_rows = [r for r in results if isinstance(r, dict)]
    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\n  Manifest saved: {manifest_path}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download all Binance USDT-M Futures klines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval", type=str, default="1h",
        choices=["5m", "15m", "1h", "4h", "1d"],
        help="Kline interval (default: 1h)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/raw",
        help="Base output directory (default: data/raw)",
    )
    parser.add_argument(
        "--pairs", nargs="+", type=str, default=None,
        help="Specific pairs to download (default: all)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List pairs without downloading",
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=MAX_CONCURRENT,
        help=f"Max parallel downloads (default: {MAX_CONCURRENT})",
    )
    args = parser.parse_args()

    asyncio.run(run_download(args))


if __name__ == "__main__":
    main()
