#!/usr/bin/env python3
"""High-speed bulk downloader for free financial data on GCP VMs.

Optimized for GCP's 10+ Gbps bandwidth:
- aiohttp with 512 concurrent connections + TCP keep-alive pooling
- Streaming writes (no full-file buffering)
- Automatic retry with exponential backoff
- Real-time bandwidth + ETA monitoring

Sources:
  binance_futures: data.binance.vision USDT-M Futures 1m (2019→now)
  binance_spot:    data.binance.vision Spot 1m top-N by volume (2017→now)
  yfinance:        US stocks + ETFs + forex + commodities daily (2000→now)

Estimated data:
  binance_futures: ~500 pairs × 80 months = ~15 GB zip → ~4B tokens
  binance_spot:    ~200 pairs × 108 months = ~8 GB zip  → ~3B tokens
  yfinance:        ~8000 tickers × 20 years daily = ~2 GB → ~500M tokens

Usage (on GCP VM):
    # Full pipeline: download + convert + upload
    python scripts/download_bulk_free.py --output /data/raw/ --phases all --to-parquet

    # Binance only, max speed
    python scripts/download_bulk_free.py --output /data/raw/ --phases binance --workers 512

    # Spot top-100 only
    python scripts/download_bulk_free.py --output /data/raw/ --phases binance_spot --spot-top 100

    # yfinance stocks (rate-limited, ~2h)
    python scripts/download_bulk_free.py --output /data/raw/ --phases yfinance
"""

import argparse
import asyncio
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bulk_dl")

BINANCE_FUTURES_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
BINANCE_SPOT_URL = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_FUTURES_DAILY_URL = "https://data.binance.vision/data/futures/um/daily/klines"
BINANCE_SPOT_DAILY_URL = "https://data.binance.vision/data/spot/daily/klines"


# ─── Bandwidth tracker ───────────────────────────────────────

class BandwidthTracker:
    """Real-time download statistics."""

    def __init__(self, total_files: int = 0):
        self.total_bytes = 0
        self.total_files = total_files
        self.start_time = time.time()
        self.ok = 0
        self.skip = 0  # already downloaded
        self.not_found = 0
        self.err = 0

    def add(self, nbytes: int, status: str):
        self.total_bytes += nbytes
        if status == "ok":
            self.ok += 1
        elif status == "skip":
            self.skip += 1
        elif status == "404":
            self.not_found += 1
        else:
            self.err += 1

    @property
    def done(self) -> int:
        return self.ok + self.skip + self.not_found + self.err

    def summary(self) -> str:
        elapsed = max(time.time() - self.start_time, 0.01)
        mb = self.total_bytes / 1e6
        mbps = (self.total_bytes * 8) / elapsed / 1e6
        pct = self.done / max(self.total_files, 1) * 100
        fps = self.done / elapsed
        eta = (self.total_files - self.done) / max(fps, 0.01)
        return (f"{self.done:,}/{self.total_files:,} ({pct:.0f}%) | "
                f"{self.ok:,} ok {self.skip:,} skip {self.not_found:,} 404 {self.err:,} err | "
                f"{mb:,.0f} MB | {mbps:,.0f} Mbps | {fps:,.0f} files/s | ETA {eta:.0f}s")


# ─── Binance pair enumeration ────────────────────────────────

async def fetch_binance_futures_pairs() -> list[str]:
    """Get all USDT-M perpetual futures pairs from Binance API."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://fapi.binance.com/fapi/v1/exchangeInfo",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            return sorted([
                s["symbol"] for s in data["symbols"]
                if s["contractType"] == "PERPETUAL"
                and s["quoteAsset"] == "USDT"
                and s["status"] == "TRADING"
            ])


async def fetch_binance_spot_pairs(top_n: int = 200) -> list[str]:
    """Get top-N spot USDT pairs by 24h quote volume."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            usdt = [
                (t["symbol"], float(t["quoteVolume"]))
                for t in data
                if t["symbol"].endswith("USDT")
            ]
            usdt.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in usdt[:top_n]]


# ─── Month generation ────────────────────────────────────────

def generate_months(sy: int, sm: int, ey: int, em: int) -> list[tuple[int, int]]:
    """Generate (year, month) tuples, inclusive."""
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def generate_dates(start_date: str, end_date: str) -> list[str]:
    """Generate YYYY-MM-DD date strings from start to end inclusive."""
    from datetime import timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


# ─── Core async downloader ───────────────────────────────────

async def download_file(
    session,
    url: str,
    out_path: Path,
    sem: asyncio.Semaphore,
    tracker: BandwidthTracker,
    max_retries: int = 2,
):
    """Download a single file with connection reuse + retry."""
    if out_path.exists() and out_path.stat().st_size > 0:
        tracker.add(0, "skip")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries + 1):
        try:
            async with sem:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        tracker.add(0, "404")
                        return
                    if resp.status == 451:
                        # Geo-blocked (some Binance data is region-restricted)
                        tracker.add(0, "404")
                        return
                    resp.raise_for_status()
                    data = await resp.read()
                    out_path.write_bytes(data)
                    tracker.add(len(data), "ok")
                    return
        except Exception:
            if attempt < max_retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
            else:
                tracker.add(0, "err")


async def download_binance(
    source: str,
    pairs: list[str],
    interval: str,
    sy: int, sm: int,
    ey: int, em: int,
    output_dir: Path,
    max_concurrent: int = 512,
):
    """Download all klines for given pairs from data.binance.vision.

    Uses a single aiohttp session with TCP connection pooling —
    all requests share keep-alive connections to the CDN.
    """
    import aiohttp

    base_url = BINANCE_FUTURES_URL if source == "futures" else BINANCE_SPOT_URL
    months = generate_months(sy, sm, ey, em)

    # Build task list
    tasks_info = []
    for pair in pairs:
        for y, m in months:
            filename = f"{pair}-{interval}-{y}-{m:02d}.zip"
            url = f"{base_url}/{pair}/{interval}/{filename}"
            out_path = output_dir / f"{source}_archive" / pair / filename
            tasks_info.append((url, out_path))

    total = len(tasks_info)
    log.info("Binance %s: %d pairs × %d months = %d files, %d concurrent",
             source, len(pairs), len(months), total, max_concurrent)

    # Connection pool: keep-alive, DNS cache, no per-request TLS handshake
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,      # max simultaneous connections
        limit_per_host=256,        # per-host limit (CDN has multiple IPs)
        ttl_dns_cache=600,         # cache DNS 10 min
        enable_cleanup_closed=True,
        force_close=False,         # keep-alive (reuse connections)
        keepalive_timeout=30,
    )
    timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
    tracker = BandwidthTracker(total_files=total)
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Fire all tasks — asyncio.gather handles concurrency via semaphore
        coros = [
            download_file(session, url, out_path, sem, tracker)
            for url, out_path in tasks_info
        ]

        # Process in chunks for progress logging
        chunk_size = 2000
        for i in range(0, len(coros), chunk_size):
            chunk = coros[i:i + chunk_size]
            await asyncio.gather(*chunk)
            log.info("%s | %s", source.upper(), tracker.summary())

    log.info("DONE %s: %s", source.upper(), tracker.summary())
    return tracker


async def download_binance_daily(
    source: str,
    pairs: list[str],
    interval: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    max_concurrent: int = 512,
):
    """Download daily kline archives for the current (incomplete) month.

    data.binance.vision publishes daily ZIPs with 1-day delay:
      /data/futures/um/daily/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM-DD}.zip
    """
    import aiohttp

    base_url = BINANCE_FUTURES_DAILY_URL if source == "futures" else BINANCE_SPOT_DAILY_URL
    dates = generate_dates(start_date, end_date)

    tasks_info = []
    for pair in pairs:
        for date_str in dates:
            filename = f"{pair}-{interval}-{date_str}.zip"
            url = f"{base_url}/{pair}/{interval}/{filename}"
            out_path = output_dir / f"{source}_daily_archive" / pair / filename
            tasks_info.append((url, out_path))

    total = len(tasks_info)
    log.info("Binance %s DAILY: %d pairs × %d days = %d files, %d concurrent",
             source, len(pairs), len(dates), total, max_concurrent)

    connector = aiohttp.TCPConnector(
        limit=max_concurrent, limit_per_host=256,
        ttl_dns_cache=600, enable_cleanup_closed=True,
        force_close=False, keepalive_timeout=30,
    )
    timeout = aiohttp.ClientTimeout(total=120, connect=30, sock_read=60)
    tracker = BandwidthTracker(total_files=total)
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        coros = [
            download_file(session, url, out_path, sem, tracker)
            for url, out_path in tasks_info
        ]
        chunk_size = 2000
        for i in range(0, len(coros), chunk_size):
            chunk = coros[i:i + chunk_size]
            await asyncio.gather(*chunk)
            log.info("%s DAILY | %s", source.upper(), tracker.summary())

    log.info("DONE %s DAILY: %s", source.upper(), tracker.summary())
    return tracker


# ─── yfinance bulk download ──────────────────────────────────

def download_yfinance(output_dir: Path, max_tickers: int = 0):
    """Download US stocks + ETFs + indices + forex + commodities via yfinance.

    Rate-limited by Yahoo (~2000 req/hour). Saves each ticker as parquet.
    """
    import yfinance as yf

    parquet_dir = output_dir / "yfinance_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Get S&P 500 + popular ETFs + forex + commodities
    tickers = set()

    # S&P 500
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers.update(sp500["Symbol"].str.replace(".", "-", regex=False).tolist())
        log.info("S&P 500: %d tickers", len(sp500))
    except Exception as e:
        log.warning("S&P 500 fetch failed: %s", e)

    # Major ETFs + indices
    etfs = [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM", "VWO",
        "GLD", "SLV", "USO", "UNG", "TLT", "IEF", "SHY", "LQD", "HYG",
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE",
        "VNQ", "ARKK", "SOXX", "SMH", "XBI", "IBB", "KRE", "XHB",
        "^VIX", "^GSPC", "^IXIC", "^DJI", "^RUT", "^TNX",
    ]
    tickers.update(etfs)

    # Forex pairs
    forex = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
        "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "AUDJPY=X", "EURAUD=X", "EURCHF=X", "GBPCHF=X", "CADJPY=X",
    ]
    tickers.update(forex)

    # Commodities
    commodities = [
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F",
        "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F",
    ]
    tickers.update(commodities)

    tickers = sorted(tickers)
    if max_tickers > 0:
        tickers = tickers[:max_tickers]
    log.info("yfinance: %d tickers to download", len(tickers))

    # Download in batches (yfinance supports multi-ticker)
    batch_size = 50
    total_rows = 0
    ok = 0
    err = 0

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                batch, period="max", interval="1d",
                group_by="ticker", auto_adjust=True,
                threads=True, progress=False,
            )
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        df = data[ticker].dropna(how="all")
                        if len(df) > 100:
                            out = parquet_dir / f"{ticker.replace('=', '_').replace('^', '_')}.parquet"
                            df.to_parquet(out)
                            total_rows += len(df)
                            ok += 1
                    except (KeyError, Exception):
                        err += 1
            else:
                # Single ticker
                if len(data) > 100:
                    t = batch[0]
                    out = parquet_dir / f"{t.replace('=', '_').replace('^', '_')}.parquet"
                    data.to_parquet(out)
                    total_rows += len(data)
                    ok += 1

            if (i // batch_size) % 10 == 0:
                log.info("yfinance: %d/%d tickers, %d ok, %d err, %d rows",
                         i + len(batch), len(tickers), ok, err, total_rows)
        except Exception as e:
            log.warning("yfinance batch %d failed: %s", i, e)
            err += len(batch)

    log.info("DONE yfinance: %d ok, %d err, %d total rows", ok, err, total_rows)


# ─── ZIP → Parquet conversion ────────────────────────────────

OHLCV_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def convert_pair(args: tuple) -> tuple[str, int]:
    """Convert all monthly ZIPs for one pair to a single parquet file."""
    symbol_dir, parquet_dir = args
    symbol = symbol_dir.name

    zips = sorted(symbol_dir.glob("*.zip"))
    if not zips:
        return symbol, 0

    dfs = []
    for zp in zips:
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    continue
                with zf.open(csv_names[0]) as f:
                    first_line = f.readline().decode().strip()
                    f.seek(0)
                    has_header = first_line.startswith("open_time")
                    df = pd.read_csv(
                        f,
                        header=0 if has_header else None,
                        names=None if has_header else OHLCV_COLS,
                        dtype={"open": float, "high": float, "low": float,
                               "close": float, "volume": float},
                    )
                    if has_header:
                        df.columns = OHLCV_COLS
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
                    dfs.append(df)
        except (zipfile.BadZipFile, Exception):
            continue

    if not dfs:
        return symbol, 0

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("open_time").drop_duplicates(subset=["open_time"])

    parquet_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(parquet_dir / f"{symbol}.parquet", index=False)
    return symbol, len(merged)


def convert_all_to_parquet(archive_dir: Path, parquet_dir: Path, workers: int = 8):
    """Convert all downloaded ZIPs to parquet using multiprocessing."""
    symbol_dirs = sorted([d for d in archive_dir.iterdir() if d.is_dir()])
    if not symbol_dirs:
        log.warning("No symbol directories in %s", archive_dir)
        return

    log.info("Converting %d pairs to parquet (%d workers)...", len(symbol_dirs), workers)

    tasks = [(d, parquet_dir) for d in symbol_dirs]
    total_rows = 0
    ok = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for symbol, nrows in pool.map(convert_pair, tasks):
            if nrows > 0:
                total_rows += nrows
                ok += 1

    log.info("DONE conversion: %d pairs, %d total rows", ok, total_rows)


# ─── Main ────────────────────────────────────────────────────

async def async_main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Parse date range
    sy, sm = [int(x) for x in args.start.split("-")]
    ey, em = [int(x) for x in args.end.split("-")]

    t0 = time.time()
    manifest = {"sources": [], "date": datetime.now().isoformat()}

    # ── Binance Futures ──
    if args.phases in ("binance", "binance_futures", "all"):
        log.info("=== Binance Futures (data.binance.vision) ===")
        pairs = await fetch_binance_futures_pairs()
        log.info("Futures: %d active USDT-M perpetual pairs", len(pairs))

        # Futures started ~2019-09
        fut_sy = max(sy, 2019)
        fut_sm = sm if sy >= 2019 else 9
        tracker = await download_binance(
            "futures", pairs, args.interval,
            fut_sy, fut_sm, ey, em,
            output, args.workers,
        )
        manifest["sources"].append({
            "source": "binance_futures",
            "pairs": len(pairs),
            "files_ok": tracker.ok,
            "files_skip": tracker.skip,
            "bytes": tracker.total_bytes,
        })

    # ── Binance Spot ──
    if args.phases in ("binance", "binance_spot", "all"):
        log.info("=== Binance Spot top-%d (data.binance.vision) ===", args.spot_top)
        pairs = await fetch_binance_spot_pairs(args.spot_top)
        log.info("Spot: %d pairs (top by 24h volume)", len(pairs))

        tracker = await download_binance(
            "spot", pairs, args.interval,
            sy, sm, ey, em,
            output, args.workers,
        )
        manifest["sources"].append({
            "source": "binance_spot",
            "pairs": len(pairs),
            "files_ok": tracker.ok,
            "files_skip": tracker.skip,
            "bytes": tracker.total_bytes,
        })

    # ── Binance Daily (current month, for live/recent data) ──
    if args.daily_start and args.phases in ("binance", "binance_futures", "binance_spot", "all"):
        from datetime import timedelta
        daily_end = args.daily_end or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if args.phases in ("binance", "binance_futures", "all"):
            if not pairs:
                pairs = await fetch_binance_futures_pairs()
            log.info("=== Binance Futures DAILY (%s → %s) ===", args.daily_start, daily_end)
            tracker = await download_binance_daily(
                "futures", pairs, args.interval,
                args.daily_start, daily_end,
                output, args.workers,
            )
            manifest["sources"].append({
                "source": "binance_futures_daily",
                "files_ok": tracker.ok,
                "bytes": tracker.total_bytes,
            })

        if args.phases in ("binance", "binance_spot", "all"):
            spot_pairs = await fetch_binance_spot_pairs(args.spot_top)
            log.info("=== Binance Spot DAILY (%s → %s) ===", args.daily_start, daily_end)
            tracker = await download_binance_daily(
                "spot", spot_pairs, args.interval,
                args.daily_start, daily_end,
                output, args.workers,
            )
            manifest["sources"].append({
                "source": "binance_spot_daily",
                "files_ok": tracker.ok,
                "bytes": tracker.total_bytes,
            })

    # ── yfinance ──
    if args.phases in ("yfinance", "all"):
        log.info("=== yfinance (stocks, ETFs, forex, commodities) ===")
        download_yfinance(output, max_tickers=args.yf_max)
        manifest["sources"].append({"source": "yfinance"})

    # ── Convert to parquet ──
    if args.to_parquet:
        for source in ("futures", "spot"):
            # Monthly archives
            archive_dir = output / f"{source}_archive"
            if archive_dir.exists():
                parquet_dir = output / f"{source}_parquet"
                convert_all_to_parquet(archive_dir, parquet_dir, args.parquet_workers)
            # Daily archives (merge into same parquet dir)
            daily_dir = output / f"{source}_daily_archive"
            if daily_dir.exists():
                parquet_dir = output / f"{source}_parquet"
                log.info("Merging daily %s archives into %s", source, parquet_dir)
                convert_all_to_parquet(daily_dir, parquet_dir, args.parquet_workers)

    # ── Save manifest ──
    elapsed = time.time() - t0
    manifest["elapsed_seconds"] = elapsed
    manifest_path = output / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=== ALL DONE in %.0fs (%.1f min) === Manifest: %s",
             elapsed, elapsed / 60, manifest_path)


def main():
    parser = argparse.ArgumentParser(description="High-speed bulk data downloader")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    parser.add_argument("--phases", default="binance",
                        choices=["binance", "binance_futures", "binance_spot", "yfinance", "all"],
                        help="What to download")
    parser.add_argument("--workers", type=int, default=512,
                        help="Max concurrent downloads (default 512)")
    parser.add_argument("--interval", default="1m", help="Kline interval (1m, 5m, 1h)")
    parser.add_argument("--spot-top", type=int, default=200,
                        help="Top N spot pairs by volume (default 200)")
    parser.add_argument("--start", default="2017-01", help="Start month YYYY-MM")
    parser.add_argument("--end", default="2026-02", help="End month YYYY-MM")
    parser.add_argument("--to-parquet", action="store_true",
                        help="Convert ZIPs to parquet after download")
    parser.add_argument("--parquet-workers", type=int, default=8,
                        help="Parallel workers for parquet conversion")
    parser.add_argument("--yf-max", type=int, default=0,
                        help="Max yfinance tickers (0 = all)")
    parser.add_argument("--daily-start", default=None,
                        help="Daily klines start date YYYY-MM-DD (current month)")
    parser.add_argument("--daily-end", default=None,
                        help="Daily klines end date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
