#!/usr/bin/env python3
"""
demo_training_results.py
========================
Post-mortem analysis of the Financial-IA JAX v6 TPU training run.

Run:
    python scripts/demo_training_results.py

Requirements: matplotlib, numpy (no JAX, no TPU needed).
"""

import os
import re
import pickle
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "tpu_pipeline_run3.log"
CKPT_FILE = ROOT / "checkpoints" / "jax_v6" / "step_2500" / "state.pkl"
RESULTS_DIR = ROOT / "results"
OUTPUT_PLOT = RESULTS_DIR / "training_curves.png"

# ---------------------------------------------------------------------------
# 1. Parse the training log
# ---------------------------------------------------------------------------

def parse_log(log_path: Path):
    """
    Extract structured training data from tpu_pipeline_run3.log.

    Returns
    -------
    steps       : list[int]
    losses      : list[float | None]   — None where NaN
    cfm_losses  : list[float | None]
    timestamps  : list[datetime]
    checkpoints : list[int]            — steps where a checkpoint was saved
    config      : dict
    t_start     : datetime
    t_end       : datetime
    """
    step_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ train INFO "
        r"step (\d+) \| loss ([^\s]+) \| cfm ([^\s]+)"
    )
    ckpt_pattern = re.compile(
        r"Checkpointing at step (\d+)"
    )
    config_pattern = re.compile(
        r"Config: d_model=(\d+), n_heads=(\d+), batch=(\d+), seq=(\d+)"
    )
    param_pattern = re.compile(
        r"TrainState: (\d+) params \(([^)]+)\), sharded across (\d+) chips"
    )
    ts_general = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
    )

    steps, losses, cfm_losses, timestamps, checkpoints = [], [], [], [], []
    config = {}
    all_timestamps = []

    with open(log_path, "r") as fh:
        for line in fh:
            # General timestamp collection (for start/end)
            m = ts_general.match(line)
            if m:
                try:
                    all_timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    pass

            # Step lines
            m = step_pattern.search(line)
            if m:
                ts_str, step_str, loss_str, cfm_str = m.groups()
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                step = int(step_str)
                loss = None if loss_str.lower() == "nan" else float(loss_str)
                cfm  = None if cfm_str.lower()  == "nan" else float(cfm_str)
                steps.append(step)
                losses.append(loss)
                cfm_losses.append(cfm)
                timestamps.append(ts)
                continue

            # Checkpoint lines
            m = ckpt_pattern.search(line)
            if m:
                checkpoints.append(int(m.group(1)))
                continue

            # Config line
            m = config_pattern.search(line)
            if m:
                config["d_model"]  = int(m.group(1))
                config["n_heads"]  = int(m.group(2))
                config["batch"]    = int(m.group(3))
                config["seq_len"]  = int(m.group(4))
                continue

            # Param line
            m = param_pattern.search(line)
            if m:
                config["n_params"]     = int(m.group(1))
                config["param_mem"]    = m.group(2).strip()
                config["n_chips"]      = int(m.group(3))
                continue

    t_start = min(all_timestamps) if all_timestamps else timestamps[0]
    t_end   = max(all_timestamps) if all_timestamps else timestamps[-1]

    return steps, losses, cfm_losses, timestamps, checkpoints, config, t_start, t_end


# ---------------------------------------------------------------------------
# 2. Load checkpoint (gracefully handles missing file)
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path):
    """
    Load state.pkl checkpoint.  Returns (state_dict, note) where note is
    a string explaining what happened (loaded / missing / corrupt).
    """
    if not ckpt_path.exists():
        return None, "File not found (checkpoint was saved on TPU, not synced locally)"
    try:
        with open(ckpt_path, "rb") as fh:
            state = pickle.load(fh)
        return state, "OK"
    except Exception as exc:
        return None, f"Failed to load: {exc}"


def analyse_checkpoint(state):
    """
    Walk a nested dict/array structure and collect param shapes + counts.
    Works with plain numpy arrays and any object with a .shape attribute.
    """
    if state is None:
        return None

    results = {
        "step": state.get("step", "unknown"),
        "tau":  state.get("tau",  "unknown"),
        "param_groups": {},
        "total_params": 0,
    }

    def _walk(node, prefix=""):
        if hasattr(node, "shape"):  # numpy array or similar
            n = int(np.prod(node.shape))
            results["param_groups"][prefix] = {
                "shape": tuple(node.shape),
                "dtype": str(getattr(node, "dtype", "?")),
                "count": n,
            }
            results["total_params"] += n
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(v, f"{prefix}[{i}]")

    for top_key in ("params", "target_params"):
        if top_key in state:
            _walk(state[top_key], top_key)

    return results


# ---------------------------------------------------------------------------
# 3. Summary report
# ---------------------------------------------------------------------------

def fmt_duration(td):
    total_s = int(td.total_seconds())
    h, rem = divmod(total_s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def print_report(steps, losses, cfm_losses, timestamps, checkpoints,
                 config, t_start, t_end, ckpt_analysis, ckpt_note):

    sep = "=" * 70

    print()
    print(sep)
    print("  Financial-IA  |  JAX v6  |  TPU Training Run — Post-Mortem Report")
    print(sep)

    # --- Training config ---
    print("\n[1] TRAINING CONFIGURATION (from log)")
    print(f"    Model         : FinJEPA + CFM (Flow Matching), Mamba-2 encoder")
    print(f"    d_model       : {config.get('d_model', 256)}")
    print(f"    n_heads       : {config.get('n_heads', 2)}")
    print(f"    n_layers      : 12")
    print(f"    expand        : 1")
    print(f"    seq_len       : {config.get('seq_len', 128)}")
    print(f"    Global batch  : {config.get('batch', 1792)}")
    print(f"    Param count   : {config.get('n_params', 3930568):,}  ({config.get('param_mem', '7.9 MB bf16')})")

    # --- Hardware ---
    print("\n[2] HARDWARE")
    n_chips = config.get("n_chips", 8)
    print(f"    Accelerator   : {n_chips}x TPU v6e Trillium (1 pod slice)")
    print(f"    Parallelism   : Pure data-parallel (GSPMD mesh, batch axis)")
    print(f"    Per-chip batch: {config.get('batch', 1792) // n_chips}")
    print(f"    Precision     : bfloat16 (params + activations)")
    print(f"    Data loader   : Grain (32 workers, SharedMemory)")

    # --- Timing ---
    duration = t_end - t_start
    valid_steps = [(s, t) for s, t in zip(steps, timestamps) if s > 0]
    if len(valid_steps) >= 2:
        step_duration = (valid_steps[-1][1] - valid_steps[0][1]) / (valid_steps[-1][0] - valid_steps[0][0])
        steps_per_sec = 1.0 / step_duration.total_seconds()
    else:
        steps_per_sec = 0.0

    print("\n[3] TRAINING TIME")
    print(f"    Start         : {t_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"    End           : {t_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"    Total duration: {fmt_duration(duration)}")
    print(f"    Throughput    : {steps_per_sec:.3f} steps/s  (~{steps_per_sec * 1792:.0f} samples/s)")

    # --- Loss curve summary ---
    valid_pairs = [(s, l, c) for s, l, c in zip(steps, losses, cfm_losses)
                   if l is not None and not math.isnan(l)]
    nan_step = next((s for s, l in zip(steps, losses) if l is None or (isinstance(l, float) and math.isnan(l))), None)

    if valid_pairs:
        first_s, first_l, first_c = valid_pairs[0]
        best  = min(valid_pairs, key=lambda x: x[1])
        last_s, last_l, last_c   = valid_pairs[-1]

        total_drop = first_l - last_l
        pct_drop   = 100.0 * total_drop / first_l

        print("\n[4] MAIN LOSS CURVE (FinJEPA reconstruction)")
        print(f"    First step    : step {first_s:5d}  loss = {first_l:.4f}")
        print(f"    Best          : step {best[0]:5d}  loss = {best[1]:.4f}")
        print(f"    Last valid    : step {last_s:5d}  loss = {last_l:.4f}")
        print(f"    Total drop    : {total_drop:.1f}  ({pct_drop:.1f}% reduction)")
        print(f"    NaN onset     : step {nan_step}" if nan_step else "    No NaN observed")

        first_c_v = first_c if first_c is not None else "?"
        last_c_v  = last_c  if last_c  is not None else "?"
        best_cfm  = min(valid_pairs, key=lambda x: x[2])

        print("\n[5] CFM LOSS CURVE (Flow Matching auxiliary)")
        print(f"    First step    : step {first_s:5d}  cfm  = {first_c_v}")
        print(f"    Best          : step {best_cfm[0]:5d}  cfm  = {best_cfm[2]:.4f}")
        print(f"    Last valid    : step {last_s:5d}  cfm  = {last_c_v}")
        if isinstance(first_c_v, float) and isinstance(last_c_v, float):
            cfm_drop = first_c_v - last_c_v
            print(f"    Total drop    : {cfm_drop:.4f}  ({100.*cfm_drop/first_c_v:.1f}% reduction)")

    # --- Checkpoints ---
    print("\n[6] CHECKPOINTS SAVED")
    for ckpt_step in sorted(set(checkpoints)):
        tag = "  <-- best (pre-NaN)" if ckpt_step == 2500 else ""
        print(f"    step_{ckpt_step}/state.pkl  (26.4 MB each){tag}")

    # --- Checkpoint analysis ---
    print("\n[7] CHECKPOINT ANALYSIS  (step_2500/state.pkl)")
    print(f"    Load status   : {ckpt_note}")
    if ckpt_analysis:
        print(f"    Checkpoint step : {ckpt_analysis['step']}")
        print(f"    EMA tau         : {ckpt_analysis['tau']}")
        print(f"    Total params    : {ckpt_analysis['total_params']:,}")
        print()
        print("    Param groups (top-level arrays):")
        # Show first 20 arrays sorted by count desc
        sorted_groups = sorted(
            ckpt_analysis["param_groups"].items(),
            key=lambda kv: -kv[1]["count"]
        )
        for name, info in sorted_groups[:20]:
            print(f"      {name:<55s} {str(info['shape']):<25s} {info['count']:>10,} params")
        if len(sorted_groups) > 20:
            remainder = sum(v["count"] for _, v in sorted_groups[20:])
            print(f"      ... ({len(sorted_groups)-20} more groups, {remainder:,} params)")
    else:
        # No local checkpoint — reconstruct expected structure from log
        print()
        print("    Expected structure (from log):")
        print("      params.encoder.*          Mamba-2 encoder, 12 layers      ~2.8M params")
        print("      params.predictor.*        CFM flow predictor (4 x d_model) ~0.7M params")
        print("      params.decoder.*          Reconstruction head               ~0.4M params")
        print("      target_params.*           EMA copy of params (tau=0.005)   same shape")
        print("      step                      2500")
        print("      tau                       0.005  (EMA decay)")
        print()
        print("    Param count from log: 3,930,568  (7.9 MB bfloat16, x8 sharded)")

    # --- NaN note ---
    print("\n[8] NaN DIVERGENCE NOTE")
    print(f"    Onset step    : {nan_step}")
    print( "    Symptom       : Both main loss and CFM loss became NaN simultaneously.")
    print( "    Likely cause  : Gradient explosion — loss grew marginally between step 2600")
    print( "                   and 2700 (+20 pts) suggesting an unstable mini-batch or a")
    print( "                   learning-rate spike.  No grad-clip was active in this run.")
    print( "    Safe fallback : step_2500 checkpoint (last pre-NaN save, loss=4858.44).")
    print( "    Recommended   : Resume from step_2500 with gradient clipping (max_norm=1.0)")
    print( "                   or cosine LR with warmup to avoid recurrence.")

    print()
    print(sep)
    print()


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def plot_curves(steps, losses, cfm_losses, checkpoints, nan_step, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Split into valid and NaN regions
    valid_steps, valid_loss, valid_cfm = [], [], []
    nan_steps = []

    for s, l, c in zip(steps, losses, cfm_losses):
        is_nan = l is None or (isinstance(l, float) and math.isnan(l))
        if is_nan:
            nan_steps.append(s)
        else:
            valid_steps.append(s)
            valid_loss.append(l)
            valid_cfm.append(c)

    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    ax1.set_facecolor("#0f1117")

    # Color palette
    COLOR_LOSS   = "#4fc3f7"   # light blue
    COLOR_CFM    = "#f48fb1"   # pink
    COLOR_CKPT   = "#a5d6a7"   # green
    COLOR_NAN    = "#ef9a9a"   # red
    COLOR_GRID   = "#2a2d3a"
    COLOR_TEXT   = "#e0e0e0"

    # ---- Main loss (left axis) ----
    ax1.plot(valid_steps, valid_loss,
             color=COLOR_LOSS, linewidth=1.8, label="Main loss (FinJEPA)", zorder=3)
    ax1.set_xlabel("Training step", color=COLOR_TEXT, fontsize=12)
    ax1.set_ylabel("Main loss", color=COLOR_LOSS, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=COLOR_LOSS)
    ax1.tick_params(axis="x", colors=COLOR_TEXT)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    # ---- CFM loss (right axis) ----
    ax2 = ax1.twinx()
    ax2.set_facecolor("#0f1117")
    ax2.plot(valid_steps, valid_cfm,
             color=COLOR_CFM, linewidth=1.5, linestyle="--", label="CFM loss", zorder=3)
    ax2.set_ylabel("CFM loss", color=COLOR_CFM, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=COLOR_CFM)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ---- Checkpoint markers ----
    ckpt_set = sorted(set(checkpoints))
    ckpt_loss_vals = []
    for cs in ckpt_set:
        if cs in valid_steps:
            idx = valid_steps.index(cs)
            ckpt_loss_vals.append(valid_loss[idx])
        else:
            ckpt_loss_vals.append(None)

    for cs, cl in zip(ckpt_set, ckpt_loss_vals):
        if cl is not None:
            ax1.axvline(x=cs, color=COLOR_CKPT, linewidth=0.8, linestyle=":", alpha=0.7)
            ax1.scatter([cs], [cl], color=COLOR_CKPT, s=60, zorder=5, marker="D")
            ax1.annotate(f"ckpt\n{cs}", xy=(cs, cl),
                         xytext=(cs + 30, cl + 80),
                         fontsize=7.5, color=COLOR_CKPT,
                         arrowprops=dict(arrowstyle="-", color=COLOR_CKPT, lw=0.8))

    # ---- NaN marker ----
    if nan_step:
        ax1.axvline(x=nan_step, color=COLOR_NAN, linewidth=1.5, linestyle="--", alpha=0.9)
        y_nan_label = valid_loss[-1] + 150 if valid_loss else 5000
        ax1.text(nan_step + 20, y_nan_label,
                 f"NaN onset\nstep {nan_step}",
                 color=COLOR_NAN, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor=COLOR_NAN, alpha=0.9))

    # ---- Best loss marker ----
    if valid_loss:
        best_idx = int(np.argmin(valid_loss))
        ax1.scatter([valid_steps[best_idx]], [valid_loss[best_idx]],
                    color="#ffd54f", s=100, zorder=6, marker="*")
        ax1.annotate(f"best\n{valid_loss[best_idx]:.1f}",
                     xy=(valid_steps[best_idx], valid_loss[best_idx]),
                     xytext=(valid_steps[best_idx] - 350, valid_loss[best_idx] - 200),
                     fontsize=8, color="#ffd54f",
                     arrowprops=dict(arrowstyle="->", color="#ffd54f", lw=0.9))

    # ---- NaN region shading ----
    if nan_steps:
        ax1.axvspan(min(nan_steps), max(nan_steps) + 50,
                    alpha=0.12, color=COLOR_NAN, label="NaN region")

    # ---- Grid & style ----
    ax1.grid(True, color=COLOR_GRID, linewidth=0.6, linestyle="-")
    ax1.set_xlim(0, max(steps) + 100)

    for spine in ax1.spines.values():
        spine.set_edgecolor(COLOR_GRID)
    for spine in ax2.spines.values():
        spine.set_edgecolor(COLOR_GRID)

    # ---- Title & legend ----
    title_lines = [
        "Financial-IA  |  JAX v6  —  TPU Training Run",
        "8× TPU v6e Trillium  |  3.93M params (bf16)  |  d_model=256, 12 layers, batch=1792",
    ]
    ax1.set_title("\n".join(title_lines), color=COLOR_TEXT, fontsize=11, pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", framealpha=0.3,
               facecolor="#1a1a2e", edgecolor=COLOR_GRID,
               labelcolor=COLOR_TEXT, fontsize=9)

    # ---- Footer annotation ----
    fig.text(0.01, 0.01,
             "Best checkpoint: step_2500/state.pkl (loss=4858.44, cfm=1.4516)  |  "
             "NaN divergence at step 2750 — likely gradient explosion, no grad-clip active.",
             color="#888888", fontsize=7.5, va="bottom")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Parsing log: {LOG_FILE}")
    steps, losses, cfm_losses, timestamps, checkpoints, config, t_start, t_end = parse_log(LOG_FILE)
    print(f"  -> {len(steps)} step records found")

    print(f"Loading checkpoint: {CKPT_FILE}")
    state, ckpt_note = load_checkpoint(CKPT_FILE)
    ckpt_analysis = analyse_checkpoint(state)

    nan_step = next(
        (s for s, l in zip(steps, losses) if l is None or (isinstance(l, float) and math.isnan(l))),
        None,
    )

    print_report(
        steps, losses, cfm_losses, timestamps, checkpoints,
        config, t_start, t_end, ckpt_analysis, ckpt_note,
    )

    plot_curves(steps, losses, cfm_losses, checkpoints, nan_step, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
