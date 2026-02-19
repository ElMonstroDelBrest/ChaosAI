"""Fin-JEPA: Financial Joint-Embedding Predictive Architecture in JAX/Flax.

Port of strate_ii/jepa.py. Combines context encoder (E_x, Mamba-2),
predictor, block masking, VICReg loss, and OT-CFM flow predictor.

Key JAX/Flax differences:
  - EMA target encoder: managed via TrainState (target_params), not in model.
    The model takes params and target_params separately.
  - Block masking: pre-computed in numpy (Grain transform), passed in batch.
  - Randomness: explicit PRNGKey threading (no implicit torch RNG).
  - No nn.Module state: pure functional — loss_fn(params, batch) -> loss.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array
from jax.random import PRNGKey

from .config import StrateIIConfig
from .encoders.mamba2_encoder import Mamba2Encoder
from .predictors.predictor import Predictor
from .predictors.flow_predictor import FlowPredictor
from .losses.vicreg import vicreg_loss


class FinJEPA(nn.Module):
    """Fin-JEPA model (functional, no EMA state).

    EMA target params are managed externally in TrainState.
    The forward pass receives target_params explicitly.
    """
    # Encoder config
    num_codes: int = 1024
    codebook_dim: int = 64
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 6
    n_heads: int = 2
    expand_factor: int = 2
    conv_kernel: int = 4
    seq_len: int = 128
    chunk_size: int = 128

    # Predictor config
    pred_hidden_dim: int = 256
    pred_n_layers: int = 2
    pred_dropout: float = 0.1
    pred_z_dim: int = 32

    # CFM config
    cfm_weight: float = 1.0
    cfm_n_steps: int = 2
    cfm_ot: bool = True

    # VICReg config
    inv_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    var_gamma: float = 1.0

    @classmethod
    def from_config(cls, config: StrateIIConfig) -> "FinJEPA":
        """Construct FinJEPA from a StrateIIConfig dataclass."""
        return cls(
            num_codes=config.embedding.num_codes,
            codebook_dim=config.embedding.codebook_dim,
            d_model=config.mamba2.d_model,
            d_state=config.mamba2.d_state,
            n_layers=config.mamba2.n_layers,
            n_heads=config.mamba2.n_heads,
            expand_factor=config.mamba2.expand_factor,
            conv_kernel=config.mamba2.conv_kernel,
            seq_len=config.embedding.seq_len,
            chunk_size=config.mamba2.chunk_size,
            pred_hidden_dim=config.predictor.hidden_dim,
            pred_n_layers=config.predictor.n_layers,
            pred_dropout=config.predictor.dropout,
            pred_z_dim=config.predictor.z_dim,
            cfm_weight=config.predictor.cfm_weight,
            cfm_n_steps=config.predictor.cfm_n_steps,
            cfm_ot=config.predictor.cfm_ot,
            inv_weight=config.vicreg.inv_weight,
            var_weight=config.vicreg.var_weight,
            cov_weight=config.vicreg.cov_weight,
            var_gamma=config.vicreg.var_gamma,
        )

    def setup(self):
        self.context_encoder = Mamba2Encoder(
            num_codes=self.num_codes,
            codebook_dim=self.codebook_dim,
            d_model=self.d_model,
            d_state=self.d_state,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            expand_factor=self.expand_factor,
            conv_kernel=self.conv_kernel,
            seq_len=self.seq_len,
            chunk_size=self.chunk_size,
            name="context_encoder",
        )

        self.predictor = Predictor(
            d_model=self.d_model,
            hidden_dim=self.pred_hidden_dim,
            n_layers=self.pred_n_layers,
            seq_len=self.seq_len,
            dropout=self.pred_dropout,
            z_dim=self.pred_z_dim,
            name="predictor",
        )

        if self.cfm_weight > 0.0:
            self.flow_predictor = FlowPredictor(
                d_model=self.d_model,
                hidden_dim=self.pred_hidden_dim,
                n_layers=self.pred_n_layers,
                seq_len=self.seq_len,
                dropout=self.pred_dropout,
                ot=self.cfm_ot,
                name="flow_predictor",
            )
        else:
            self.flow_predictor = None

        self.output_proj = nn.Dense(self.codebook_dim, name="output_proj")

    def __call__(
        self,
        batch: dict[str, Array],
        target_params: dict,
        key: PRNGKey,
        deterministic: bool = False,
    ) -> dict[str, Array]:
        """Forward pass: encode -> predict -> VICReg loss.

        Args:
            batch: dict with keys:
                - token_indices: (B, S) int64
                - weekend_mask: (B, S) float32
                - block_mask: (B, S) bool (pre-computed in Grain)
                - exo_clock: (B, S, 2) float32 or None
                - target_positions: (B, N_tgt) int64 (pre-computed)
                - target_mask: (B, N_tgt) bool (valid targets, excl. padding)
            target_params: FrozenDict of target encoder params (EMA).
            key: PRNGKey for noise sampling.
            deterministic: If True, disable dropout.

        Returns:
            dict with loss components and diagnostics.
        """
        token_indices = batch["token_indices"]
        weekend_mask = batch.get("weekend_mask")
        block_mask = batch.get("block_mask")
        exo_clock = batch.get("exo_clock")
        target_positions = batch["target_positions"]
        target_mask = batch["target_mask"]

        B, S = token_indices.shape
        key_z, key_cfm = jax.random.split(key)

        # 1. Context encoder: sees full sequence with [MASK] at target positions
        h_x = self.context_encoder(
            token_indices,
            weekend_mask=weekend_mask,
            block_mask=block_mask,
            exo_clock=exo_clock,
        )  # (B, S, d_model)

        # 2. Target encoder: same architecture, EMA weights, NO masking
        # Apply context_encoder with target_params (extract encoder subset)
        # target_params is the full model params dict; we need context_encoder's subset.
        # When target_params is None (init), fall back to self-encoding.
        if target_params is not None:
            encoder_target_params = target_params["context_encoder"]
        else:
            encoder_target_params = None

        if encoder_target_params is not None:
            h_y = jax.lax.stop_gradient(
                self.context_encoder.apply(
                    {"params": encoder_target_params},
                    token_indices,
                    weekend_mask=weekend_mask,
                    block_mask=None,  # Target sees everything
                    exo_clock=exo_clock,
                )
            )
        else:
            # During init: just use context encoder directly
            h_y = jax.lax.stop_gradient(
                self.context_encoder(
                    token_indices,
                    weekend_mask=weekend_mask,
                    block_mask=None,
                    exo_clock=exo_clock,
                )
            )  # (B, S, d_model)

        # 3. Predict target representations with stochastic noise z
        N_tgt = target_positions.shape[1]
        z = jax.random.normal(key_z, (B, N_tgt, self.pred_z_dim), dtype=h_x.dtype)
        h_y_pred = self.predictor(
            h_x, target_positions, z=z, deterministic=deterministic
        )  # (B, N_tgt, d_model)

        # 4. Gather true target representations from target encoder
        # target_positions: (B, N_tgt) -> index into h_y: (B, S, d_model)
        h_y_tgt = jax.vmap(lambda h, idx: h[idx])(h_y, target_positions)  # (B, N_tgt, d_model)

        # 5. Flatten valid targets for VICReg (exclude padding)
        h_pred_flat = h_y_pred[target_mask]  # (N_valid, d_model)
        h_tgt_flat = jax.lax.stop_gradient(h_y_tgt[target_mask])  # (N_valid, d_model)

        # 6. VICReg loss
        loss_dict = vicreg_loss(
            h_pred_flat, h_tgt_flat,
            inv_weight=self.inv_weight,
            var_weight=self.var_weight,
            cov_weight=self.cov_weight,
            var_gamma=self.var_gamma,
        )
        total_loss = loss_dict["total"]

        # 7. CFM loss (Phase D) — train v_theta to transport N(0,I) -> h_y_tgt
        cfm_loss = jnp.float32(0.0)
        if self.flow_predictor is not None and self.cfm_weight > 0.0:
            h_y_tgt_stopped = jax.lax.stop_gradient(h_y_tgt)
            v_pred, v_tgt = self.flow_predictor(
                h_x, target_positions, h_y_tgt_stopped,
                key=key_cfm, deterministic=deterministic,
            )
            # Mask padding positions before MSE
            v_pred_flat = v_pred[target_mask]
            v_tgt_flat = v_tgt[target_mask]
            cfm_loss = jnp.mean((v_pred_flat - v_tgt_flat) ** 2)
            total_loss = total_loss + self.cfm_weight * cfm_loss

        return {
            "loss": total_loss,
            "invariance": loss_dict["invariance"],
            "variance": loss_dict["variance"],
            "covariance": loss_dict["covariance"],
            "cfm_loss": cfm_loss,
            "mask_ratio": jnp.mean(block_mask.astype(jnp.float32)) if block_mask is not None else jnp.float32(0.0),
            "n_targets": jnp.sum(target_mask.astype(jnp.float32)) / B,
        }

    def generate_futures(
        self,
        params: dict,
        target_params: dict,
        token_indices: Array,
        weekend_mask: Array | None,
        target_positions: Array,
        key: PRNGKey,
        n_samples: int = 16,
        exo_clock: Array | None = None,
    ) -> Array:
        """Generate N stochastic future trajectories in latent space.

        Args:
            params: Model params (context encoder + predictor + flow).
            target_params: EMA target encoder params.
            token_indices: (B, S) context token indices.
            weekend_mask: (B, S) or None.
            target_positions: (B, N_tgt) positions to predict.
            key: PRNGKey.
            n_samples: Number of divergent futures.
            exo_clock: (B, S, 2) or None.

        Returns:
            (N, B, N_tgt, d_model) — N latent trajectories.
        """
        # Encode context
        h_x = self.apply(
            {"params": params},
            token_indices,
            weekend_mask=weekend_mask,
            exo_clock=exo_clock,
            method=self.context_encoder,
        )

        B, N_tgt = target_positions.shape
        futures = []

        if self.flow_predictor is not None:
            for i in range(n_samples):
                sample_key = jax.random.fold_in(key, i)
                h_pred = self.flow_predictor.sample(
                    params["flow_predictor"],
                    h_x, target_positions,
                    key=sample_key,
                    n_steps=self.cfm_n_steps,
                )
                futures.append(h_pred)
        else:
            for i in range(n_samples):
                sample_key = jax.random.fold_in(key, i)
                z = jax.random.normal(sample_key, (B, N_tgt, self.pred_z_dim), dtype=h_x.dtype)
                h_pred = self.apply(
                    {"params": params},
                    h_x, target_positions, z, True,
                    method=self.predictor,
                )
                futures.append(h_pred)

        return jnp.stack(futures)  # (N, B, N_tgt, d_model)
