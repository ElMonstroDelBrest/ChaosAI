#!/usr/bin/env python3
"""Download Binance Futures funding rates, Open Interest, and Long/Short ratio.

Endpoints:
  - /fapi/v1/fundingRate         (funding rate history, weight=1)
  - /futures/data/openInterestHist (OI history, weight=1)
  - /futures/data/globalLongShortAccountRatio (weight=1)

All 432 USDT-M pairs, full history, saved as parquet per pair.
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

BASE_URL = "https://fapi.binance.com"
MAX_CONCURRENT = 5
WEIGHT_LIMIT = 2400
PROACTIVE_THRESHOLD = 0.80

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


async def get_all_pairs(session: aiohttp.ClientSession) -> list[str]:
    """Fetch all active USDT-M perpetual pairs."""
    async with session.get(f"{BASE_URL}/fapi/v1/exchangeInfo") as resp:
        data = await resp.json()
    pairs = [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT"
        and s["contractType"] == "PERPETUAL"
        and s["status"] == "TRADING"
    ]
    return sorted(pairs)


async def throttle_if_needed(resp_headers: dict):
    """Check Binance weight headers and throttle proactively."""
    used = int(resp_headers.get("X-MBX-USED-WEIGHT-1M", "0"))
    if used > WEIGHT_LIMIT * PROACTIVE_THRESHOLD:
        wait = 5 + (used - WEIGHT_LIMIT * PROACTIVE_THRESHOLD) / 100
        log.info(f"Throttle: {used}/{WEIGHT_LIMIT} weight, pausing {wait:.1f}s")
        await asyncio.sleep(wait)


async def download_funding_rates(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbol: str,
    output_dir: Path,
):
    """Download full funding rate history for a symbol."""
    out_file = output_dir / f"{symbol}.parquet"
    if out_file.exists():
        return

    all_rows = []
    start_time = 1_500_000_000_000  # ~2017
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    while start_time < end_time:
        async with sem:
            params = {"symbol": symbol, "startTime": start_time, "limit": 1000}
            try:
                async with session.get(f"{BASE_URL}/fapi/v1/fundingRate", params=params) as resp:
                    await throttle_if_needed(resp.headers)
                    if resp.status == 429:
                        await asyncio.sleep(30)
                        continue
                    data = await resp.json()
            except Exception as e:
                log.warning(f"{symbol} funding error: {e}")
                await asyncio.sleep(5)
                continue

        if not data:
            break

        all_rows.extend(data)
        start_time = data[-1]["fundingTime"] + 1

    if all_rows:
        df = pd.DataFrame(all_rows)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        df["markPrice"] = df["markPrice"].astype(float)
        df.to_parquet(out_file, compression="zstd")


async def download_oi_history(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbol: str,
    output_dir: Path,
    period: str = "5m",
):
    """Download Open Interest history for a symbol."""
    out_file = output_dir / f"{symbol}.parquet"
    if out_file.exists():
        return

    all_rows = []
    start_time = 1_600_000_000_000  # ~2020
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    while start_time < end_time:
        async with sem:
            params = {
                "symbol": symbol, "period": period,
                "startTime": start_time, "limit": 500,
            }
            try:
                async with session.get(
                    f"{BASE_URL}/futures/data/openInterestHist", params=params
                ) as resp:
                    await throttle_if_needed(resp.headers)
                    if resp.status == 429:
                        await asyncio.sleep(30)
                        continue
                    if resp.status != 200:
                        break
                    data = await resp.json()
            except Exception as e:
                log.warning(f"{symbol} OI error: {e}")
                await asyncio.sleep(5)
                continue

        if not data:
            break

        all_rows.extend(data)
        start_time = data[-1]["timestamp"] + 1

    if all_rows:
        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["sumOpenInterest", "sumOpenInterestValue"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df.to_parquet(out_file, compression="zstd")


async def download_long_short_ratio(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbol: str,
    output_dir: Path,
    period: str = "5m",
):
    """Download global long/short account ratio history."""
    out_file = output_dir / f"{symbol}.parquet"
    if out_file.exists():
        return

    all_rows = []
    start_time = 1_600_000_000_000
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    while start_time < end_time:
        async with sem:
            params = {
                "symbol": symbol, "period": period,
                "startTime": start_time, "limit": 500,
            }
            try:
                async with session.get(
                    f"{BASE_URL}/futures/data/globalLongShortAccountRatio",
                    params=params,
                ) as resp:
                    await throttle_if_needed(resp.headers)
                    if resp.status == 429:
                        await asyncio.sleep(30)
                        continue
                    if resp.status != 200:
                        break
                    data = await resp.json()
            except Exception as e:
                log.warning(f"{symbol} LS ratio error: {e}")
                await asyncio.sleep(5)
                continue

        if not data:
            break

        all_rows.extend(data)
        start_time = data[-1]["timestamp"] + 1

    if all_rows:
        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["longAccount", "shortAccount", "longShortRatio"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df.to_parquet(out_file, compression="zstd")


async def main():
    parser = argparse.ArgumentParser(description="Download Binance Futures funding/OI/LS data")
    parser.add_argument("--output_dir", default="data/raw", help="Base output directory")
    parser.add_argument("--pairs", nargs="+", help="Limit to specific pairs")
    args = parser.parse_args()

    base = Path(args.output_dir)
    funding_dir = base / "funding_rates"
    oi_dir = base / "open_interest"
    ls_dir = base / "long_short_ratio"
    for d in [funding_dir, oi_dir, ls_dir]:
        d.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        pairs = args.pairs or await get_all_pairs(session)
        log.info(f"Downloading funding/OI/LS for {len(pairs)} pairs")

        # Funding rates
        log.info("=== Funding Rates ===")
        tasks = [download_funding_rates(session, sem, p, funding_dir) for p in pairs]
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Funding"):
            await coro

        # Open Interest
        log.info("=== Open Interest (5m) ===")
        tasks = [download_oi_history(session, sem, p, oi_dir) for p in pairs]
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="OI"):
            await coro

        # Long/Short ratio
        log.info("=== Long/Short Ratio (5m) ===")
        tasks = [download_long_short_ratio(session, sem, p, ls_dir) for p in pairs]
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LS Ratio"):
            await coro

    # Summary
    for name, d in [("Funding", funding_dir), ("OI", oi_dir), ("LS Ratio", ls_dir)]:
        files = list(d.glob("*.parquet"))
        total_mb = sum(f.stat().st_size for f in files) / 1e6
        log.info(f"{name}: {len(files)} pairs, {total_mb:.1f} MB")

    log.info("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
