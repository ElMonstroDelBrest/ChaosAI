"""Mamba-2 block with selective scan and weekend gating.

Uses fused CUDA kernels from mamba-ssm when available, falls back to
pure-PyTorch sequential scan otherwise.

Weekend gating: delta *= (1 - is_weekend)
  When weekend: delta=0 → exp(A*0)=I → h_t = h_{t-1} (state frozen).
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Try to import fused selective scan kernel
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_FUSED_SCAN = True
except ImportError:
    HAS_FUSED_SCAN = False

# Try to import fused causal conv1d kernel
try:
    from causal_conv1d import causal_conv1d_fn
    HAS_FUSED_CONV = True
except ImportError:
    HAS_FUSED_CONV = False


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left padding."""

    def __init__(self, d_inner: int, kernel_size: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size, groups=d_inner)

    def forward(self, x: Tensor) -> Tensor:
        """(B, D, L) -> (B, D, L)."""
        if HAS_FUSED_CONV:
            # causal_conv1d_fn expects (B, D, L) and weight (D, kernel_size)
            return causal_conv1d_fn(
                x=x,
                weight=self.conv.weight.squeeze(1),  # (D, 1, K) -> (D, K)
                bias=self.conv.bias,
            )
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


def selective_scan_slow(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    weekend_mask: Tensor | None = None,
) -> Tensor:
    """Pure-PyTorch sequential selective scan (fallback)."""
    B_batch, L, D_inner = x.shape
    n_heads = A.shape[0]
    d_state = A.shape[1]
    head_dim = D_inner // n_heads

    x_heads = x.view(B_batch, L, n_heads, head_dim)
    dt = F.softplus(dt)

    if weekend_mask is not None:
        gate = 1.0 - weekend_mask.unsqueeze(-1)
        dt = dt * gate

    h = torch.zeros(B_batch, n_heads, d_state, head_dim, device=x.device, dtype=x.dtype)
    outputs = []

    for t in range(L):
        dt_t = dt[:, t, :]
        B_t = B[:, t, :, :]
        C_t = C[:, t, :, :]
        x_t = x_heads[:, t, :, :]

        A_bar = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
        B_bar = dt_t.unsqueeze(-1) * B_t

        h = A_bar.unsqueeze(-1) * h + B_bar.unsqueeze(-1) * x_t.unsqueeze(2)
        y_t = torch.einsum("bns,bnsd->bnd", C_t, h)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)
    y = y.reshape(B_batch, L, D_inner)
    return y


def selective_scan_fused(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    weekend_mask: Tensor | None = None,
) -> Tensor:
    """Fused CUDA selective scan from mamba-ssm.

    mamba-ssm selective_scan_fn expects:
        u:     (B, D, L)        input
        delta: (B, D, L)        time step (post-softplus)
        A:     (D, N)           SSM state matrix
        B:     (B, N, L)        input matrix
        C:     (B, N, L)        output matrix
        D:     (D,) or None     skip connection
        z:     (B, D, L) or None  gate

    Our shapes:
        x:  (B, L, D_inner)
        dt: (B, L, n_heads) — needs expanding to D_inner
        A:  (n_heads, d_state) — needs expanding to D_inner
        B:  (B, L, n_heads, d_state) — needs reshaping
        C:  (B, L, n_heads, d_state) — needs reshaping
    """
    B_batch, L, D_inner = x.shape
    n_heads = A.shape[0]
    d_state = A.shape[1]
    head_dim = D_inner // n_heads

    # Fused selective_scan_cuda only supports float32 — upcast everything from bf16
    input_dtype = x.dtype
    x = x.float()

    # Apply softplus to dt
    dt = F.softplus(dt).float()  # (B, L, n_heads)

    # Weekend gating: zero out delta on weekends
    if weekend_mask is not None:
        gate = 1.0 - weekend_mask.float().unsqueeze(-1)  # (B, L, 1)
        dt = dt * gate

    # Expand dt from (B, L, n_heads) to (B, L, D_inner) by repeating per head
    dt = dt.unsqueeze(-1).expand(-1, -1, -1, head_dim).reshape(B_batch, L, D_inner)

    # Expand A from (n_heads, d_state) to (D_inner, d_state)
    A_expanded = A.unsqueeze(1).expand(-1, head_dim, -1).reshape(D_inner, d_state)

    # Reshape B, C from (B, L, n_heads, d_state) to (B, d_state, L)
    # For multi-head: we need to handle heads. selective_scan_fn treats D independently,
    # so expanding B/C per head works since each head-dim uses the same B/C for that head.
    # B: (B, L, n_heads, d_state) -> expand to (B, L, D_inner, d_state) isn't right...
    # Actually selective_scan_fn expects B: (B, N, L) where N=d_state, applied uniformly to all D.
    # But we have per-head B. We need to run per-head or flatten.

    # Per-head approach: reshape everything to treat heads as batch dim
    # x: (B, L, n_heads, head_dim) -> (B*n_heads, head_dim, L)
    x_heads = x.view(B_batch, L, n_heads, head_dim).permute(0, 2, 3, 1)  # (B, H, hd, L)
    x_flat = x_heads.reshape(B_batch * n_heads, head_dim, L)

    # dt: (B, L, n_heads) -> (B, H, L) -> (B*H, head_dim_repeated, L)
    dt_heads = dt.view(B_batch, L, n_heads, head_dim).permute(0, 2, 3, 1)
    dt_flat = dt_heads.reshape(B_batch * n_heads, head_dim, L)

    # A: (n_heads, d_state) -> repeat for each batch -> (B*H, head_dim, d_state)
    # selective_scan_fn expects A: (D, N) where D=head_dim
    A_per_head = A.float().unsqueeze(1).expand(-1, head_dim, -1)  # (H, hd, N)

    # B: (B, L, H, N) -> (B, H, N, L) -> (B*H, N, L)
    B_flat = B.float().permute(0, 2, 3, 1).reshape(B_batch * n_heads, d_state, L)

    # C: (B, L, H, N) -> (B, H, N, L) -> (B*H, N, L)
    C_flat = C.float().permute(0, 2, 3, 1).reshape(B_batch * n_heads, d_state, L)

    # Run fused scan for ALL heads in a single kernel call.
    # selective_scan_fn broadcasts A: (D, N) over the batch dimension.
    # We have B*H batches with x_flat already interleaved as [b0h0, b0h1, ..., b1h0, ...].
    # Reorder so consecutive batch entries share the same A (group by head):
    #   x_flat: (B*H, hd, L) interleaved → (H, B, hd, L) → (H*B, hd, L)
    # Then run per-head with contiguous batches (single kernel, no Python loop).

    # Reorder from interleaved (b0h0,b0h1,...,b1h0,...) to grouped (h0b0,h0b1,...,h1b0,...)
    x_grouped = x_flat.view(B_batch, n_heads, head_dim, L).permute(1, 0, 2, 3).reshape(n_heads * B_batch, head_dim, L)
    dt_grouped = dt_flat.view(B_batch, n_heads, head_dim, L).permute(1, 0, 2, 3).reshape(n_heads * B_batch, head_dim, L)
    B_grouped = B_flat.view(B_batch, n_heads, d_state, L).permute(1, 0, 2, 3).reshape(n_heads * B_batch, d_state, L)
    C_grouped = C_flat.view(B_batch, n_heads, d_state, L).permute(1, 0, 2, 3).reshape(n_heads * B_batch, d_state, L)

    # A_per_head: (H, hd, N) → tile each head's A for B batches → (H*B, hd, N)
    # selective_scan_fn expects A: (D, N) broadcast over batch, but with grouped layout
    # each contiguous block of B entries uses the same A[h].
    # We must tile: (H, hd, N) → (H, 1, hd, N) → (H, B, hd, N) → (H*B, hd, N)
    A_tiled = A_per_head.unsqueeze(1).expand(-1, B_batch, -1, -1).reshape(n_heads * B_batch, head_dim, d_state)

    # Single fused kernel call with H*B batches
    # NOTE: selective_scan_fn expects A: (D, N) not (B, D, N).
    # It broadcasts A over batch. Since different batches need different A (per-head),
    # we split into n_heads calls but with contiguous B-sized batches (no striding).
    outputs = []
    for h in range(n_heads):
        start = h * B_batch
        end = start + B_batch
        y_h = selective_scan_fn(
            x_grouped[start:end],        # (B, hd, L) — contiguous
            dt_grouped[start:end],       # (B, hd, L)
            A_per_head[h],               # (hd, N)
            B_grouped[start:end],        # (B, N, L)
            C_grouped[start:end],        # (B, N, L)
            D=None,
            z=None,
            delta_softplus=False,        # already applied
            return_last_state=False,
        )
        outputs.append(y_h)  # (B, hd, L)

    # Stack heads: (H, B, hd, L) → (B, H, hd, L) → (B, L, D_inner)
    y = torch.stack(outputs, dim=0)  # (H, B, hd, L)
    y = y.permute(1, 3, 0, 2).reshape(B_batch, L, D_inner)
    return y.to(input_dtype)


def selective_scan(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    weekend_mask: Tensor | None = None,
) -> Tensor:
    """Selective scan with automatic backend selection."""
    if HAS_FUSED_SCAN and x.is_cuda:
        return selective_scan_fused(x, dt, A, B, C, weekend_mask)
    return selective_scan_slow(x, dt, A, B, C, weekend_mask)


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with selective scan and weekend gating.

    Architecture:
        x → LayerNorm → Linear → split(x_branch, z_gate, B, C, dt)
        x_branch → CausalConv1d → SiLU → selective_scan → y
        y = y * SiLU(z_gate)
        y → Linear → + residual
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        n_heads: int = 2,
        expand_factor: int = 2,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.d_state = d_state
        self.n_heads = n_heads
        self.head_dim = self.d_inner // n_heads

        assert self.d_inner % n_heads == 0, "d_inner must be divisible by n_heads"

        self.norm = nn.LayerNorm(d_model)

        # Input projection: x → x_branch, z_gate, B, C, dt
        # Sizes: d_inner + d_inner + n_heads*d_state + n_heads*d_state + n_heads
        self.in_proj_size = (
            self.d_inner        # x_branch
            + self.d_inner      # z_gate
            + n_heads * d_state # B
            + n_heads * d_state # C
            + n_heads           # dt
        )
        self.in_proj = nn.Linear(d_model, self.in_proj_size, bias=False)

        # Causal conv on x_branch
        self.conv = CausalConv1d(self.d_inner, conv_kernel)

        # SSM parameter A (log-space, negative, learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(n_heads, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: Tensor, weekend_mask: Tensor | None = None) -> Tensor:
        """Forward pass with optional weekend gating.

        Args:
            x: (B, L, d_model) input sequence.
            weekend_mask: (B, L) float {0.0, 1.0} where 1.0 = weekend.

        Returns:
            (B, L, d_model) output with residual connection.
        """
        residual = x
        x = self.norm(x)

        # Project and split
        proj = self.in_proj(x)  # (B, L, in_proj_size)

        idx = 0
        x_branch = proj[..., idx:idx + self.d_inner]
        idx += self.d_inner
        z_gate = proj[..., idx:idx + self.d_inner]
        idx += self.d_inner
        B_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        C_proj = proj[..., idx:idx + self.n_heads * self.d_state]
        idx += self.n_heads * self.d_state
        dt = proj[..., idx:idx + self.n_heads]

        # Reshape B, C: (B, L, n_heads * d_state) → (B, L, n_heads, d_state)
        B_proj = B_proj.view(*B_proj.shape[:-1], self.n_heads, self.d_state)
        C_proj = C_proj.view(*C_proj.shape[:-1], self.n_heads, self.d_state)

        # Causal conv on x_branch: (B, L, D) → (B, D, L) → conv → (B, L, D)
        x_branch = self.conv(x_branch.transpose(1, 2)).transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM: A in negative log-space
        A = -torch.exp(self.A_log)  # (n_heads, d_state)

        # Selective scan with weekend gating
        y = selective_scan(x_branch, dt, A, B_proj, C_proj, weekend_mask)

        # Gated output
        y = y * F.silu(z_gate)

        # Output projection + residual
        return self.out_proj(y) + residual
