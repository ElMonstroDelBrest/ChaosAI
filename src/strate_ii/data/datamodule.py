"""Lightning DataModule for Strate II — optimized for high-throughput GPU training."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .token_dataset import TokenSequenceDataset, SyntheticTokenDataset


class StrateIIDataModule(pl.LightningDataModule):
    """DataModule for Strate II pre-training.

    Uses pre-tokenized sequences from Strate I, or synthetic data for dev/test.

    H100 optimizations:
        - pin_memory=True: async CPU→GPU transfer via DMA
        - persistent_workers=True: avoid worker respawn overhead
        - prefetch_factor=4: pipeline data loading ahead of GPU
        - drop_last=True: consistent batch sizes for Tensor Core efficiency

    Args:
        token_dir: Directory containing .pt token files.
        seq_len: Sequence length.
        num_codes: Codebook size (for synthetic).
        batch_size: Batch size.
        val_split: Fraction for validation.
        num_workers: DataLoader workers.
        prefetch_factor: Batches pre-loaded per worker.
        synthetic: If True, use synthetic data instead of real tokens.
        num_synthetic: Number of synthetic sequences.
    """

    def __init__(
        self,
        token_dir: str = "data/tokens/",
        seq_len: int = 64,
        num_codes: int = 1024,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        synthetic: bool = False,
        num_synthetic: int = 512,
    ):
        super().__init__()
        self.token_dir = token_dir
        self.seq_len = seq_len
        self.num_codes = num_codes
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.synthetic = synthetic
        self.num_synthetic = num_synthetic

    def setup(self, stage: str | None = None):
        if self.synthetic:
            full = SyntheticTokenDataset(
                num_sequences=self.num_synthetic,
                seq_len=self.seq_len,
                num_codes=self.num_codes,
            )
        else:
            full = TokenSequenceDataset(
                token_dir=self.token_dir,
                seq_len=self.seq_len,
            )

        n_val = int(len(full) * self.val_split)
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(
            full, [n_train, n_val],
            generator=__import__("torch").Generator().manual_seed(42),
        )

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        use_workers = self.num_workers > 0
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=use_workers,
            prefetch_factor=self.prefetch_factor if use_workers else None,
            drop_last=shuffle,  # drop_last for train only
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)
