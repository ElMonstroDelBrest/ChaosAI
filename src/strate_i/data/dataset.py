import os
from glob import glob

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import PatchTransform


def _load_ohlcv(path: str) -> Tensor | None:
    """Load OHLCV from .pt, .npy, or .parquet → (T, 5) tensor."""
    try:
        if path.endswith(".pt"):
            return torch.load(path, weights_only=True)
        elif path.endswith(".npy"):
            return torch.from_numpy(np.load(path)).float()
        elif path.endswith(".parquet"):
            import pyarrow.parquet as pq
            table = pq.read_table(path, columns=["open", "high", "low", "close", "volume"])
            return torch.from_numpy(table.to_pandas().values).float()
        return None
    except Exception:
        return None


class OHLCVPatchDataset(Dataset):
    """Loads OHLCV files (.pt/.npy/.parquet), applies log-returns + patching.

    Supports multiple data directories (semicolon-separated in data_dir).
    Example: data_dir="data/raw_1m_parquet;data/ohlcv_stocks_daily"
    """

    def __init__(self, data_dir: str, patch_length: int = 16, stride: int = 16):
        super().__init__()
        self.transform = PatchTransform(patch_length, stride)
        self.patches = self._load(data_dir)

    def _load(self, data_dir: str) -> Tensor:
        # Support multiple dirs separated by semicolons
        dirs = [d.strip() for d in data_dir.split(";") if d.strip()]
        file_paths = []
        for d in dirs:
            file_paths.extend(sorted(glob(os.path.join(d, "*.pt"))))
            file_paths.extend(sorted(glob(os.path.join(d, "*.npy"))))
            file_paths.extend(sorted(glob(os.path.join(d, "*.parquet"))))

        if not file_paths:
            raise FileNotFoundError(f"No .pt/.npy/.parquet files found in {data_dir}")

        all_patches = []
        for path in file_paths:
            raw = _load_ohlcv(path)
            if raw is None or raw.ndim != 2 or raw.shape[1] != 5:
                continue
            patches = self.transform(raw)  # (N, L, C)
            if patches.size(0) > 0:
                all_patches.append(patches)

        return torch.cat(all_patches, dim=0)

    def __len__(self) -> int:
        return self.patches.size(0)

    def __getitem__(self, idx: int) -> Tensor:
        return self.patches[idx]
