"""Download recent 1min candles for out-of-sample evaluation.

Downloads the last N days of 1-minute candles from Binance Futures API
for a subset of major pairs, saves as .pt files (OHLCV, same format as training data).

Usage:
    PYTHONPATH=. python scripts/download_eval_data.py --days 1 --output_dir data/ohlcv_eval/
"""

import argparse
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dl_eval")

# Top 50 pairs by volume (Binance Futures)
TOP_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "NEARUSDT",
    "AAVEUSDT", "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT",
    "FILUSDT", "INJUSDT", "TIAUSDT", "JUPUSDT", "WIFUSDT",
    "ONDOUSDT", "SEIUSDT", "STXUSDT", "RUNEUSDT", "ENAUSDT",
    "MKRUSDT", "FETUSDT", "GALAUSDT", "IMXUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "ICPUSDT", "ALGOUSDT", "VETUSDT",
    "FTMUSDT", "THETAUSDT", "GRTUSDT", "SNXUSDT", "CRVUSDT",
    "LDOUSDT", "PENDLEUSDT", "WLDUSDT", "ORDIUSDT", "PEPEUSDT",
]


async def download_pair(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
    sem: asyncio.Semaphore,
) -> list:
    """Download all 1min klines for a pair in the given time range."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    all_klines = []
    current = start_ms

    while current < end_ms:
        async with sem:
            params = {
                "symbol": symbol,
                "interval": "1m",
                "startTime": current,
                "endTime": end_ms,
                "limit": 1500,
            }
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 30))
                        log.warning("%s: rate limited, waiting %ds", symbol, retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    if resp.status != 200:
                        log.warning("%s: HTTP %d, skipping", symbol, resp.status)
                        break
                    data = await resp.json()
                    if not data:
                        break
                    all_klines.extend(data)
                    last_ts = data[-1][0]
                    if last_ts <= current:
                        break  # no progress
                    current = last_ts + 60_000
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("%s: %s, retrying", symbol, e)
                await asyncio.sleep(2)
                continue
        await asyncio.sleep(0.05)  # light rate limit

    return all_klines


def klines_to_tensor(klines: list) -> torch.Tensor:
    """Convert Binance klines to (N, 5) OHLCV float32 tensor."""
    rows = []
    for k in klines:
        # k = [open_time, open, high, low, close, volume, ...]
        rows.append([float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
    return torch.tensor(rows, dtype=torch.float32)


async def main(pairs: list[str], output_dir: str, days: int):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    end_ms = int(now.timestamp() * 1000)
    start_ms = int((now - timedelta(days=days)).timestamp() * 1000)

    log.info(
        "Downloading %d pairs, last %d day(s): %s → %s",
        len(pairs), days,
        datetime.utcfromtimestamp(start_ms / 1000).isoformat(),
        datetime.utcfromtimestamp(end_ms / 1000).isoformat(),
    )

    sem = asyncio.Semaphore(5)
    ok, fail = 0, 0

    async with aiohttp.ClientSession() as session:
        for i, symbol in enumerate(pairs):
            klines = await download_pair(session, symbol, start_ms, end_ms, sem)
            if klines:
                tensor = klines_to_tensor(klines)
                torch.save(tensor, out / f"{symbol}.pt")
                log.info("[%d/%d] %s: %d candles (%.0f min)", i + 1, len(pairs), symbol, tensor.shape[0], tensor.shape[0])
                ok += 1
            else:
                log.warning("[%d/%d] %s: no data", i + 1, len(pairs), symbol)
                fail += 1

    log.info("Done: %d OK, %d failed, output: %s", ok, fail, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download eval data from Binance")
    parser.add_argument("--output_dir", default="data/ohlcv_eval/")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--pairs", nargs="+", default=TOP_PAIRS)
    args = parser.parse_args()
    asyncio.run(main(args.pairs, args.output_dir, args.days))
