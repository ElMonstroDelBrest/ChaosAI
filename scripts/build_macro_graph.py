"""Build macro context from FRED, COT, and alternative signal files.

Reads .npy signal files from multiple directories, aligns temporally,
z-scores each signal, and outputs:
  - macro_context.npz: {signals: (T, N), signal_names: [...], timestamps: [...]}
  - correlation_graph.npz: {adj_matrix: (N_assets, N_assets), asset_names: [...]}

Input format: each .npy file is a 1D or 2D array:
  - 1D: (T,) — single signal, filename is signal name
  - 2D: (T, 2) — column 0 = unix timestamp (seconds), column 1 = value

If timestamps are not embedded in the file, assumes 1h frequency starting
from a reference date (2020-01-01 UTC).

Usage:
    PYTHONPATH=. python scripts/build_macro_graph.py \
        --signal_dirs data/fred_data/ data/cot_data/ data/alt_signals/ \
        --output_dir data/macro/ \
        --freq 1h

    # With GCS paths (download first):
    gsutil -m cp -r gs://fin-ia-eu/data/fred_data/ data/
    gsutil -m cp -r gs://fin-ia-eu/data/cot_data/ data/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("macro_graph")

REFERENCE_START = pd.Timestamp("2020-01-01", tz="UTC")


def load_signal(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a .npy or .pt signal file.

    Returns:
        values: 1D array of signal values
        timestamps: 1D array of unix timestamps (seconds), or None if not embedded
    """
    suffix = path.suffix.lower()
    if suffix == ".pt":
        import torch
        data = torch.load(path, weights_only=True).numpy()
    else:
        data = np.load(path)
    if data.ndim == 1:
        return data.astype(np.float64), None
    elif data.ndim == 2 and data.shape[1] == 2:
        return data[:, 1].astype(np.float64), data[:, 0].astype(np.float64)
    elif data.ndim == 2 and data.shape[1] > 2:
        # Multi-column: treat column 0 as timestamp, rest as separate signals
        return data[:, 1:].astype(np.float64), data[:, 0].astype(np.float64)
    else:
        log.warning("Unexpected shape %s in %s — skipping", data.shape, path.name)
        return np.array([]), None


def signal_to_series(values: np.ndarray, timestamps: np.ndarray | None,
                     signal_name: str) -> pd.DataFrame:
    """Convert signal values + optional timestamps to a DatetimeIndex DataFrame."""
    if timestamps is not None:
        idx = pd.to_datetime(timestamps, unit="s", utc=True)
    else:
        idx = pd.date_range(start=REFERENCE_START, periods=len(values), freq="h")

    if values.ndim == 1:
        return pd.DataFrame({signal_name: values}, index=idx)
    else:
        # Multi-column signal
        cols = {f"{signal_name}_{i}": values[:, i] for i in range(values.shape[1])}
        return pd.DataFrame(cols, index=idx)


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a series, handling constant series."""
    mu = s.mean()
    sigma = s.std()
    if sigma < 1e-10:
        return s * 0.0
    return (s - mu) / sigma


def build_macro_context(
    signal_dirs: list[str],
    output_dir: str,
    freq: str = "1h",
    corr_window: int = 168,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all signals
    all_series = []
    for signal_dir in signal_dirs:
        dir_path = Path(signal_dir)
        if not dir_path.exists():
            log.warning("Signal dir not found: %s — skipping", signal_dir)
            continue

        npy_files = sorted(list(dir_path.glob("*.npy")) + list(dir_path.glob("*.pt")))
        log.info("Loading %d signals from %s", len(npy_files), signal_dir)

        for npy_file in npy_files:
            values, timestamps = load_signal(npy_file)
            if len(values) == 0:
                continue
            signal_name = npy_file.stem
            df = signal_to_series(values, timestamps, signal_name)
            # Deduplicate timestamps (keep last value for each timestamp)
            df = df[~df.index.duplicated(keep="last")]
            all_series.append(df)

    if not all_series:
        log.error("No signals loaded from any directory")
        return

    log.info("Loaded %d signal sources, aligning temporally...", len(all_series))

    # Merge all signals on a common DatetimeIndex via pd.concat (O(n) vs O(n²) iterative join)
    merged = pd.concat(all_series, axis=1, join="outer")
    # Deduplicate columns (some signals may share names across dirs)
    merged = merged.loc[:, ~merged.columns.duplicated()]

    log.info("Pre-resample: %d timesteps × %d columns", len(merged), len(merged.columns))

    # Resample to target frequency (forward-fill for low-frequency signals)
    merged = merged.resample(freq).ffill()
    # Fill any remaining NaN (beginning of series) with column mean
    col_means = merged.mean()
    merged = merged.fillna(col_means)
    # Drop columns that are still all-NaN (no data at all)
    merged = merged.dropna(axis=1, how="all")
    # Drop rows that are all-NaN
    merged = merged.dropna(how="all")

    log.info("Merged signals: %d timesteps × %d signals", len(merged), len(merged.columns))

    # Z-score each signal
    for col in merged.columns:
        merged[col] = zscore_series(merged[col])

    # Build output arrays
    signal_names = list(merged.columns)
    signals = merged.values.astype(np.float32)
    timestamps = merged.index.astype(np.int64) // 10**9  # UTC seconds

    # Save macro context
    macro_path = output_path / "macro_context.npz"
    np.savez_compressed(
        macro_path,
        signals=signals,
        signal_names=np.array(signal_names, dtype=object),
        timestamps=timestamps,
    )
    log.info("Saved macro context: %s (%s)", macro_path, signals.shape)

    # Rolling correlation matrix for cross-asset graph
    # Skip rolling corr for large N (>200 cols → 40K+ elements per timestamp, very slow)
    if len(merged.columns) > 1 and len(merged.columns) <= 200 and len(merged) > corr_window:
        log.info("Computing rolling correlation (window=%d)...", corr_window)
        rolling_corr = merged.rolling(window=corr_window, min_periods=corr_window // 2).corr()
        # Take the last valid correlation matrix as the adjacency matrix
        last_valid_idx = merged.index[-1]
        try:
            adj_matrix = rolling_corr.loc[last_valid_idx].values.astype(np.float32)
            # Reshape: rolling corr produces (N*N,) at each timestamp
            n_signals = len(signal_names)
            adj_matrix = adj_matrix.reshape(n_signals, n_signals)
            # Set diagonal to 0 (no self-loops)
            np.fill_diagonal(adj_matrix, 0.0)
            # Replace NaN with 0
            adj_matrix = np.nan_to_num(adj_matrix, nan=0.0)
        except Exception as e:
            log.warning("Rolling correlation failed: %s — using static corr", e)
            adj_matrix = merged.corr().values.astype(np.float32)
            np.fill_diagonal(adj_matrix, 0.0)
            adj_matrix = np.nan_to_num(adj_matrix, nan=0.0)

        corr_path = output_path / "correlation_graph.npz"
        np.savez_compressed(
            corr_path,
            adj_matrix=adj_matrix,
            asset_names=np.array(signal_names, dtype=object),
        )
        log.info("Saved correlation graph: %s (%s)", corr_path, adj_matrix.shape)

    # Save metadata
    meta = {
        "n_signals": len(signal_names),
        "n_timesteps": len(merged),
        "freq": freq,
        "corr_window": corr_window,
        "signal_names": signal_names,
        "date_range": [str(merged.index[0]), str(merged.index[-1])],
    }
    meta_path = output_path / "macro_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata: %s", meta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build macro context from signal files")
    parser.add_argument("--signal_dirs", nargs="+", required=True,
                        help="Directories containing .npy signal files")
    parser.add_argument("--output_dir", type=str, default="data/macro/")
    parser.add_argument("--freq", type=str, default="1h",
                        help="Target frequency for resampling (default: 1h)")
    parser.add_argument("--corr_window", type=int, default=168,
                        help="Rolling correlation window in freq units (default: 168 = 1 week)")
    args = parser.parse_args()

    build_macro_context(
        signal_dirs=args.signal_dirs,
        output_dir=args.output_dir,
        freq=args.freq,
        corr_window=args.corr_window,
    )
