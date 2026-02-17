"""Token sequence dataset for Strate II.

Loads pre-tokenized sequences (from pretokenize.py) as {token_indices, weekend_mask}.
Handles sequences shorter than seq_len via zero-padding.
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path


class TokenSequenceDataset(Dataset):
    """Dataset of pre-tokenized sequences for Fin-JEPA.

    Each file is a dict with:
        - token_indices: (S,) int64 token indices
        - weekend_mask: (S,) float32 {0.0, 1.0}

    Sequences shorter than seq_len are zero-padded.
    Sequences longer than seq_len are truncated.

    Args:
        token_dir: Directory containing .pt token files.
        seq_len: Expected sequence length.
    """

    def __init__(self, token_dir: str, seq_len: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.token_dir = Path(token_dir)

        # Load all token files
        self.sequences = []
        if self.token_dir.exists():
            for pt_file in sorted(self.token_dir.glob("*.pt")):
                data = torch.load(pt_file, weights_only=True)
                self.sequences.append(data)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        data = self.sequences[idx]
        tokens = data["token_indices"]
        mask = data["weekend_mask"]

        L = len(tokens)
        if L < self.seq_len:
            # Zero-pad shorter sequences
            tokens = F.pad(tokens, (0, self.seq_len - L), value=0)
            mask = F.pad(mask, (0, self.seq_len - L), value=0.0)

        return {
            "token_indices": tokens[:self.seq_len],
            "weekend_mask": mask[:self.seq_len],
        }


class SyntheticTokenDataset(Dataset):
    """Synthetic dataset for development and testing.

    Generates random token sequences with realistic weekend patterns
    (5 trading days + 2 weekend days, repeating).

    Args:
        num_sequences: Number of sequences to generate.
        seq_len: Length of each sequence.
        num_codes: Codebook vocabulary size.
    """

    def __init__(
        self,
        num_sequences: int = 256,
        seq_len: int = 64,
        num_codes: int = 1024,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.num_codes = num_codes

        # Pre-generate all data
        self.token_indices = torch.randint(0, num_codes, (num_sequences, seq_len))
        self.weekend_masks = self._generate_weekend_masks(num_sequences, seq_len)

    @staticmethod
    def _generate_weekend_masks(num_sequences: int, seq_len: int) -> Tensor:
        """Generate cyclic weekend masks: 5 trading + 2 weekend, repeating."""
        masks = torch.zeros(num_sequences, seq_len)
        for i in range(seq_len):
            day_in_week = i % 7
            if day_in_week >= 5:  # Saturday=5, Sunday=6
                masks[:, i] = 1.0
        return masks

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "token_indices": self.token_indices[idx],
            "weekend_mask": self.weekend_masks[idx],
        }
