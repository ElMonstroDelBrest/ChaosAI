#!/bin/bash
###############################################################################
# Financial-IA — Bulk Download Binance Futures Klines
#
# Downloads from data.binance.vision (static CDN, no rate limits!)
# Massively parallel: 30 concurrent downloads.
#
# Usage:
#   chmod +x scripts/download_bulk_binance.sh
#   ./scripts/download_bulk_binance.sh              # All pairs, 1h
#   ./scripts/download_bulk_binance.sh 5m           # All pairs, 5m
#   ./scripts/download_bulk_binance.sh 1h 50        # 1h, 50 concurrent
###############################################################################

set -euo pipefail

INTERVAL="${1:-1h}"
MAX_PARALLEL="${2:-30}"
BASE_URL="https://data.binance.vision/data/futures/um/monthly/klines"
OUTPUT_DIR="data/raw/${INTERVAL}"
TEMP_DIR="/tmp/binance_bulk"

mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

echo "============================================"
echo " Binance Bulk Download (data.binance.vision)"
echo " Interval: $INTERVAL"
echo " Parallel: $MAX_PARALLEL"
echo " Output:   $OUTPUT_DIR/"
echo "============================================"
echo

# ---------------------------------------------------------------------------
# 1. Get list of all USDT-M perpetual symbols
# ---------------------------------------------------------------------------

echo "[1/3] Fetching symbol list..."

# Try Binance API first (works outside US)
SYMBOLS_JSON=$(curl -s "https://fapi.binance.com/fapi/v1/exchangeInfo" 2>/dev/null || true)

if [ -n "$SYMBOLS_JSON" ] && ! echo "$SYMBOLS_JSON" | grep -q '"code"'; then
    # API works — parse JSON
    SYMBOLS=$(echo "$SYMBOLS_JSON" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for s in data.get('symbols', []):
    if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING':
        print(s['symbol'])
" 2>/dev/null | sort)
else
    # API blocked (US IP) — use S3 bucket listing
    echo "  API blocked, using S3 bucket listing..."
    S3_XML=$(curl -s "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?prefix=data/futures/um/monthly/klines/&delimiter=/" 2>/dev/null || true)

    if echo "$S3_XML" | grep -q 'CommonPrefixes'; then
        SYMBOLS=$(echo "$S3_XML" | grep -oP 'klines/\K[A-Z0-9]+(?=/)' | sort -u | grep 'USDT$')
    else
        echo "  ERROR: Could not fetch symbol list from any source"
        exit 1
    fi
fi

NUM_SYMBOLS=$(echo "$SYMBOLS" | wc -l)
echo "  Found $NUM_SYMBOLS symbols"

# ---------------------------------------------------------------------------
# 2. Generate download URLs (all months from 2019-09 to now)
# ---------------------------------------------------------------------------

echo "[2/3] Generating download list..."

CURRENT_YEAR=$(date +%Y)
CURRENT_MONTH=$(date +%m)
URL_FILE="$TEMP_DIR/urls.txt"
> "$URL_FILE"

for SYMBOL in $SYMBOLS; do
    PARQUET_FILE="${OUTPUT_DIR}/${SYMBOL}.parquet"
    # Skip if already processed
    if [ -f "$PARQUET_FILE" ]; then
        continue
    fi

    for YEAR in $(seq 2019 "$CURRENT_YEAR"); do
        START_MONTH=1
        END_MONTH=12

        if [ "$YEAR" -eq 2019 ]; then
            START_MONTH=9  # Binance Futures launched Sept 2019
        fi
        if [ "$YEAR" -eq "$CURRENT_YEAR" ]; then
            END_MONTH=$((CURRENT_MONTH - 1))  # Current month not yet complete
            [ "$END_MONTH" -lt 1 ] && continue
        fi

        for MONTH in $(seq "$START_MONTH" "$END_MONTH"); do
            MONTH_PAD=$(printf "%02d" "$MONTH")
            ZIP_NAME="${SYMBOL}-${INTERVAL}-${YEAR}-${MONTH_PAD}.zip"
            URL="${BASE_URL}/${SYMBOL}/${INTERVAL}/${ZIP_NAME}"
            echo "$URL $SYMBOL $ZIP_NAME" >> "$URL_FILE"
        done
    done
done

NUM_URLS=$(wc -l < "$URL_FILE")
echo "  $NUM_URLS ZIP files to download"

# ---------------------------------------------------------------------------
# 3. Download + extract + merge to parquet
# ---------------------------------------------------------------------------

echo "[3/3] Downloading with $MAX_PARALLEL parallel connections..."

# Create per-symbol directories
awk '{print $2}' "$URL_FILE" | sort -u | while read SYM; do
    mkdir -p "$TEMP_DIR/$SYM"
done

# Download all ZIPs in parallel using xargs + inline bash
# Each line: URL SYMBOL ZIP_NAME
awk '{print $1, $2, $3}' "$URL_FILE" | \
    xargs -P "$MAX_PARALLEL" -L 1 bash -c '
    URL="$1"; SYMBOL="$2"; ZIP_NAME="$3"
    TEMP_DIR="'"$TEMP_DIR"'"
    ZIP_PATH="$TEMP_DIR/$SYMBOL/$ZIP_NAME"
    [ -f "$ZIP_PATH" ] && exit 0
    HTTP_CODE=$(curl -s -o "$ZIP_PATH" -w "%{http_code}" "$URL" 2>/dev/null)
    if [ "$HTTP_CODE" != "200" ]; then
        rm -f "$ZIP_PATH"
    fi
' _

DOWNLOADED=$(find "$TEMP_DIR" -name "*.zip" | wc -l)
echo "  Downloads complete. $DOWNLOADED ZIP files downloaded."
echo "  Merging CSVs to parquet..."

# Merge CSVs per symbol into parquet
python3 - "$TEMP_DIR" "$OUTPUT_DIR" "$SYMBOLS" << 'PYEOF'
import sys, os, zipfile, io
from pathlib import Path

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("  pyarrow not available, saving as CSV instead")

temp_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

symbols_done = 0
symbols_total = len(list(temp_dir.iterdir()))

for symbol_dir in sorted(temp_dir.iterdir()):
    if not symbol_dir.is_dir():
        continue
    symbol = symbol_dir.name
    out_path = output_dir / f"{symbol}.parquet"
    if out_path.exists():
        continue

    all_dfs = []
    for zip_path in sorted(symbol_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv"):
                        with zf.open(name) as f:
                            df = pd.read_csv(f, header=None, names=COLUMNS)
                            all_dfs.append(df)
        except (zipfile.BadZipFile, Exception):
            continue

    if not all_dfs:
        continue

    try:
        merged = pd.concat(all_dfs, ignore_index=True)
        # Convert timestamps to numeric first (they may be read as strings)
        merged["open_time"] = pd.to_numeric(merged["open_time"], errors="coerce")
        merged["close_time"] = pd.to_numeric(merged["close_time"], errors="coerce")
        merged = merged.dropna(subset=["open_time"])
        merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        merged["open_time"] = pd.to_datetime(merged["open_time"].astype("int64"), unit="ms")
        merged["close_time"] = pd.to_datetime(merged["close_time"].astype("int64"), unit="ms")

        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_volume", "taker_buy_quote_volume"]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

        if HAS_PARQUET:
            pq.write_table(pa.Table.from_pandas(merged), out_path)
        else:
            merged.to_csv(str(out_path).replace(".parquet", ".csv"), index=False)

        symbols_done += 1
        candles = len(merged)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [{symbols_done}] {symbol}: {candles:,} candles, {size_mb:.1f} MB")
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        continue

print(f"\nDone: {symbols_done} symbols processed to {output_dir}/")
PYEOF

echo
echo "============================================"
echo " Bulk download complete!"
echo " Data: $OUTPUT_DIR/"
echo " Size: $(du -sh "$OUTPUT_DIR/" 2>/dev/null | cut -f1)"
echo "============================================"
