# Reproducibility

This document gives the minimum setup needed to reproduce the numbers in the
[main README](../README.md) or to re-run a single component of the pipeline on
a machine that is not a TPU.

## Environments

Two independent Python environments; pick according to the hardware you have.

### CPU / GPU (PyTorch side — validation, small-scale experiments)

```bash
uv venv && source .venv/bin/activate
uv sync                 # core deps (PyTorch, mamba-ssm, gymnasium)
uv sync --extra dev     # + pytest, pytest-cov
export PYTHONPATH=$PWD  # required — src/ uses absolute imports from repo root
```

Python 3.11+ required. CUDA 12.1+ recommended for the `mamba-ssm` custom kernels;
a pure-PyTorch chunked scan fallback is available when Triton/CUDA is absent.

### TPU (JAX side — full training)

See `scripts/setup_tpu_vm.sh`. JAX/Flax/Optax/Grain/Orbax are installed
separately on the TPU VM (not in `pyproject.toml`) because the pins are
hardware-specific.

```bash
pip install 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install pytorch-lightning 'jax[tpu]' flax optax grain orbax-checkpoint diffrax
pip install tensorflow-cpu array_record dacite tslearn
```

A patch to Orbax 0.11.x is required for JAX 0.6.x compat — see the block at the
end of this file.

## Smoke Test (5 minutes, CPU or GPU)

A minimal one-batch forward+backward validation that the pipeline is wired end-to-end:

```bash
# 1. Download a tiny slice of OHLCV (~50 MB)
python scripts/download_bulk_free.py \
    --source binance_futures --interval 1h \
    --symbols BTCUSDT,ETHUSDT --output data/smoke/

# 2. Train Strate I tokenizer for 100 steps
python scripts/train_strate_i.py \
    --config configs/strate_i.yaml \
    --max_steps 100 --output checkpoints/smoke_strate_i.pt

# 3. Pre-tokenize
python scripts/pretokenize_to_arrayrecord.py \
    --checkpoint checkpoints/smoke_strate_i.pt \
    --data_dir data/smoke/ --output_dir data/smoke_ar/ --seq_len 64

# 4. Run one Strate II training step (PyTorch side)
pytest tests/test_jepa.py::test_forward -v

# 5. Verify Multiverse Crossing math on random embeddings
pytest tests/test_multiverse.py -v
```

If all five steps pass, the pipeline is installed correctly. Running the full
838M-token training requires a TPU v6e-8 or larger (~75 min end-to-end).

## Repro targets

| Result in README | Config | Compute | Wall time |
|---|---|---|---|
| Strate II loss 2,003 | `configs/scaling/v6e_multi.yaml` | v6e-8 | ~75 min |
| Strate II loss 908 (v6.1) | `configs/scaling/v6e_38m_v3.yaml` | v6e-8 | ~95 min |
| Sharpe 2.78 (Multiverse) | `scripts/run_multiverse_crossing.py` | v6e-8 | ~15 min |
| Lyapunov −0.73 | same as above | — | — |

Seeds are fixed in each config (`seed: 42`). Deviations greater than ±5% on
Sharpe / loss likely indicate a data-pipeline regression — check
`results/multiverse_crossing_30u.json` for the exact per-asset scores we obtained.

## Known platform caveats

- **Python 3.14**: `multiprocessing` is unstable with some PyTorch 2.6 + DataLoader
  patterns; set `num_workers=0` on CPU-side validation runs if you see
  `SIGKILL` at step 0 with leaked semaphores.
- **Orbax + JAX 0.6.x**: in `replica_slices.py:99`, replace
  `with jax.sharding.set_mesh(mesh):` with `if True:` — `set_mesh()` no longer
  returns a context manager in 0.6.x.
- **GCS egress**: always co-locate TPU zone and GCS bucket in the same region
  (`europe-west4-*`). Inter-region transfer is ~$0.08/GB per Grain batch load.

## Artifact locations

- Checkpoints (small, ≤270 MB): tracked in `checkpoints/`
- Large artifacts (data, embeddings, RL buffer): `gs://fin-ia-eu/`, cleanup via
  `scripts/trc_data_manager.sh cleanup --force`
- Training logs: `tb_logs/` (TensorBoard) and `results/` (JSON + markdown)

## Contact

Open a GitHub issue with the output of `python scripts/env_report.py` attached
if a step above fails on your platform.
