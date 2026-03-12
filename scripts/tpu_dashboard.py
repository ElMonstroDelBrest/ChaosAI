#!/usr/bin/env python3
"""ChaosAI TPU Dashboard — single-file TUI for TPU training lifecycle.

Usage:
    source .venv/bin/activate && python scripts/tpu_dashboard.py
"""

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# ── TPU connection wrapper ───────────────────────────────────────────────────

@dataclass
class TPUConfig:
    name: str = "fin-ia-v6e"
    zone: str = "europe-west4-a"
    project: str = "financial-ai-487700"
    gcs: str = "gs://fin-ia-eu"
    ssh_key: str = "~/.ssh/google_compute_engine"
    user: str = "daniel"
    remote_dir: str = "~/Financial_IA"
    version: str = "v2-alpha-tpuv6e"
    tpu_type: str = "v6e-8"


class TPU:
    def __init__(self, cfg: TPUConfig | None = None):
        self.cfg = cfg or TPUConfig()
        self._ip: str | None = None

    # ── IP resolution ────────────────────────────────────────────────────

    def get_ip(self) -> str | None:
        """Get external IP via gcloud describe."""
        rc, out = self.gcloud(
            f"compute tpus tpu-vm describe {self.cfg.name} "
            f"--zone={self.cfg.zone} --format='get(networkEndpoints[0].accessConfig.externalIp)'"
        )
        if rc == 0 and out.strip():
            self._ip = out.strip()
            return self._ip
        return None

    def _resolve_ip(self) -> str | None:
        """Try cached IP first (fast SSH ping), fallback to gcloud describe."""
        if self._ip:
            rc = subprocess.call(
                ["ssh", "-i", os.path.expanduser(self.cfg.ssh_key),
                 "-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=no",
                 "-o", "BatchMode=yes",
                 f"{self.cfg.user}@{self._ip}", "true"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if rc == 0:
                return self._ip
        return self.get_ip()

    # ── Command execution ────────────────────────────────────────────────

    def ssh(self, cmd: str, timeout: int = 120) -> tuple[int, str]:
        """Run a command on the TPU VM via SSH. Returns (returncode, stdout)."""
        ip = self._resolve_ip()
        if not ip:
            return 1, "TPU unreachable"
        full_cmd = [
            "ssh", "-i", os.path.expanduser(self.cfg.ssh_key),
            "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            f"{self.cfg.user}@{ip}",
            f"cd {self.cfg.remote_dir} 2>/dev/null; {cmd}",
        ]
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 1, f"SSH timeout ({timeout}s)"
        except Exception as e:
            return 1, str(e)

    def ssh_stream(self, cmd: str) -> subprocess.Popen:
        """SSH with live stdout streaming."""
        ip = self._resolve_ip()
        assert ip, "TPU unreachable"
        full_cmd = [
            "ssh", "-i", os.path.expanduser(self.cfg.ssh_key),
            "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
            f"{self.cfg.user}@{ip}",
            f"cd {self.cfg.remote_dir} 2>/dev/null; {cmd}",
        ]
        return subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def scp_upload(self) -> tuple[int, str]:
        """Upload src/, scripts/, configs/ to TPU VM."""
        ip = self._resolve_ip()
        if not ip:
            return 1, "TPU unreachable"
        dest = f"{self.cfg.user}@{ip}:{self.cfg.remote_dir}/"
        full_cmd = [
            "scp", "-r", "-i", os.path.expanduser(self.cfg.ssh_key),
            "-o", "StrictHostKeyChecking=no",
            "src", "scripts", "configs", "pyproject.toml",
            dest,
        ]
        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 1, "SCP timeout"
        except Exception as e:
            return 1, str(e)

    def gcloud(self, subcmd: str, timeout: int = 120) -> tuple[int, str]:
        """Run a gcloud command."""
        full_cmd = f"gcloud {subcmd} --project={self.cfg.project}"
        try:
            r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return 1, f"gcloud timeout ({timeout}s)"
        except Exception as e:
            return 1, str(e)

    def vm_state(self) -> str:
        """Return VM state: READY / PREEMPTED / CREATING / NOT_FOUND / error."""
        rc, out = self.gcloud(
            f"compute tpus tpu-vm describe {self.cfg.name} "
            f"--zone={self.cfg.zone} --format='get(state)'"
        )
        if rc != 0:
            if "NOT_FOUND" in out or "was not found" in out:
                return "NOT_FOUND"
            return f"ERROR: {out.strip()[:80]}"
        return out.strip() or "UNKNOWN"

    def is_reachable(self) -> bool:
        ip = self._resolve_ip()
        return ip is not None


# ── Strate II config table ───────────────────────────────────────────────────

STRATE2_CONFIGS = [
    ("v6e_xs",    "configs/scaling/v6e_xs.yaml",    "~3M",   "arrayrecord_multi", "Smoke test"),
    ("v6e_26m",   "configs/scaling/v6e_26m.yaml",   "~26M",  "arrayrecord_1m_p1", "Chinchilla 1-epoch"),
    ("v6e_multi", "configs/scaling/v6e_multi.yaml",  "~22M",  "arrayrecord_multi", "Best run (Chinchilla)"),
    ("v6e_s",     "configs/scaling/v6e_s.yaml",      "~50M",  "arrayrecord_multi", "Quick"),
    ("v6e_m",     "configs/scaling/v6e_m.yaml",      "~300M", "arrayrecord_multi", "Main scale-up"),
    ("v6e_l",     "configs/scaling/v6e_l.yaml",      "~1B",   "arrayrecord_multi", "Large (remat)"),
]

RL_MODES = [
    ("a", "All-Weather DQN (Ape-X vmap)",  "scripts/train_strate_iv_allweather.py",
     "data/rl_buffer/", "train_allweather.log",
     [("total_steps", "1000000"), ("n_envs", "256")]),
    ("b", "Cross-Sectional DQN V2",        "scripts/train_cross_sectional.py",
     "data/rl_buffer_v2/", "train_cross_sectional.log",
     [("total_steps", "500000"), ("n_portfolios", "64")]),
    ("c", "TD-MPC2 Continuous",            "scripts/train_tdmpc2_allweather.py",
     "data/rl_buffer_v2/", "train_tdmpc2.log",
     [("total_steps", "100000")]),
]

# XLA flags for training env
LIBTPU_INIT_ARGS = (
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_tpu_spmd_rng_bit_generator_unsafe=true "
    "--xla_tpu_enable_experimental_fusion_cost_model=true"
)

# ── Helper functions ─────────────────────────────────────────────────────────

def wait_key(msg: str = "Press Enter to continue..."):
    console.print(f"\n[dim]{msg}[/]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def banner():
    console.print(Panel(
        "[bold cyan]ChaosAI TPU Dashboard[/]\n"
        "[dim]Financial-IA — World Model Training Manager[/]",
        border_style="cyan",
    ))


def print_menu():
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold green", width=5)
    table.add_column()
    items = [
        ("[1]", "Status          VM state, devices, processes, disk"),
        ("[2]", "Start VM        Create TPU, auto-setup env"),
        ("[3]", "Train Strate II Select config, auto-stage data, launch"),
        ("[4]", "Train RL        DQN / Cross-Sectional / TD-MPC2"),
        ("[5]", "Monitor         Live log tail + parsed metrics"),
        ("[6]", "Data            GCS status, stage, sync, cleanup"),
        ("[7]", "Checkpoints     List, backup, sync GCS <-> Drive"),
        ("[8]", "Stop VM         Delete TPU (with confirm)"),
        ("[9]", "SSH             Interactive SSH shell"),
        ("[q]", "Quit"),
    ]
    for key, desc in items:
        table.add_row(key, desc)
    console.print(table)


# ── [1] Status ───────────────────────────────────────────────────────────────

def screen_status(tpu: TPU):
    console.print("\n[bold]Checking TPU status...[/]")
    state = tpu.vm_state()

    rows: list[tuple[str, str]] = [("VM State", state)]

    if state == "READY":
        ip = tpu.get_ip()
        rows.append(("External IP", ip or "unknown"))

        # JAX devices
        rc, out = tpu.ssh("source .venv_tpu/bin/activate 2>/dev/null; python3 -c \"import jax; ds=jax.devices(); print(f'{len(ds)} {ds[0].device_kind} chips')\"", timeout=30)
        rows.append(("JAX Devices", out.strip() if rc == 0 else "[red]unavailable[/]"))

        # Running processes
        rc, out = tpu.ssh("pgrep -af 'python3.*(train|run_training)' || echo 'none'", timeout=10)
        procs = out.strip()
        if procs == "none" or not procs:
            rows.append(("Training", "[dim]no training running[/]"))
        else:
            for line in procs.splitlines()[:3]:
                rows.append(("Process", line.strip()[:80]))

        # Disk usage
        rc, out = tpu.ssh("df -h / | tail -1 | awk '{print $3\"/\"$2\" (\"$5\" used)\"}'", timeout=10)
        rows.append(("Disk", out.strip() if rc == 0 else "?"))

        # Data dirs
        rc, out = tpu.ssh("ls -d data/*/ 2>/dev/null | head -10", timeout=10)
        if rc == 0 and out.strip():
            rows.append(("Data dirs", out.strip().replace("\n", ", ")))
        else:
            rows.append(("Data dirs", "[dim]none[/]"))

    elif state == "NOT_FOUND":
        rows.append(("", "[yellow]TPU VM does not exist. Use [2] Start VM to create.[/]"))
    elif state == "PREEMPTED":
        rows.append(("", "[red]VM was preempted. Use [2] Start VM to recreate.[/]"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold", width=14)
    table.add_column()
    for k, v in rows:
        table.add_row(k, v)

    console.print(Panel(table, title=f"[bold]{tpu.cfg.name}[/] ({tpu.cfg.tpu_type})", border_style="blue"))
    wait_key()


# ── [2] Start VM ─────────────────────────────────────────────────────────────

def screen_start_vm(tpu: TPU):
    console.print("\n[bold]Start TPU VM[/]")

    state = tpu.vm_state()
    if state == "READY":
        console.print(f"[green]TPU {tpu.cfg.name} already exists and is READY.[/]")
        choice = Prompt.ask("Action", choices=["reuse", "recreate", "cancel"], default="reuse")
        if choice == "cancel":
            return
        if choice == "recreate":
            console.print("[yellow]Deleting existing TPU...[/]")
            rc, out = tpu.gcloud(
                f"compute tpus tpu-vm delete {tpu.cfg.name} "
                f"--zone={tpu.cfg.zone} --quiet",
                timeout=300,
            )
            if rc != 0:
                console.print(f"[red]Delete failed: {out}[/]")
                wait_key()
                return
            console.print("[green]Deleted.[/]")
            state = "NOT_FOUND"
        else:
            # reuse — just upload code and verify
            _upload_and_verify(tpu)
            return

    if state in ("NOT_FOUND", "PREEMPTED"):
        if state == "PREEMPTED":
            console.print("[yellow]VM was preempted. Deleting stale entry...[/]")
            tpu.gcloud(
                f"compute tpus tpu-vm delete {tpu.cfg.name} "
                f"--zone={tpu.cfg.zone} --quiet",
                timeout=300,
            )

        console.print(f"[cyan]Creating {tpu.cfg.tpu_type} (preemptible) in {tpu.cfg.zone}...[/]")
        rc, out = tpu.gcloud(
            f"compute tpus tpu-vm create {tpu.cfg.name} "
            f"--zone={tpu.cfg.zone} "
            f"--accelerator-type={tpu.cfg.tpu_type} "
            f"--version={tpu.cfg.version} "
            f"--preemptible",
            timeout=600,
        )
        if rc != 0:
            console.print(f"[red]Create failed:\n{out}[/]")
            wait_key()
            return

        # Poll until READY
        console.print("[dim]Waiting for VM to become READY...[/]")
        for _ in range(30):
            time.sleep(10)
            s = tpu.vm_state()
            console.print(f"  State: {s}")
            if s == "READY":
                break
        else:
            console.print("[red]Timeout waiting for READY state.[/]")
            wait_key()
            return

        console.print("[green]TPU VM created.[/]")

    _upload_and_verify(tpu)


def _upload_and_verify(tpu: TPU):
    """Upload code, setup env, verify JAX."""
    console.print("\n[cyan]Uploading code...[/]")
    rc, out = tpu.scp_upload()
    if rc != 0:
        console.print(f"[red]SCP failed: {out}[/]")
        wait_key()
        return

    console.print("[cyan]Setting up environment (this may take a few minutes)...[/]")
    setup_script = """
set -euo pipefail
REPO_DIR="$HOME/Financial_IA"
cd "$REPO_DIR"

VENV="$REPO_DIR/.venv_tpu"
if [ ! -f "$VENV/bin/activate" ]; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install -q -U pip setuptools wheel
pip install -q -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -q -U flax optax orbax-checkpoint diffrax array-record grain-nightly tensorflow dacite pyyaml tqdm

python3 -c "
import jax
ds = jax.devices()
n = len(ds)
print(f'{n} {ds[0].device_kind} chips on {ds[0].platform}')
assert n >= 4, f'Expected >=4 chips, got {n}'
print('JAX verification: OK')
"
"""
    rc, out = tpu.ssh(setup_script, timeout=600)
    if rc != 0:
        console.print(f"[red]Setup failed:\n{out[-500:]}[/]")
    else:
        # Show last few lines
        for line in out.strip().splitlines()[-5:]:
            console.print(f"  {line}")
        console.print("[green]VM ready.[/]")
    wait_key()


# ── [3] Train Strate II ──────────────────────────────────────────────────────

def screen_train_strate2(tpu: TPU):
    console.print("\n[bold]Train Strate II — Config Selection[/]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="bold green", width=3)
    table.add_column("Config")
    table.add_column("Params")
    table.add_column("Data dir")
    table.add_column("Notes")
    for i, (name, path, params, data, notes) in enumerate(STRATE2_CONFIGS, 1):
        table.add_row(str(i), name, params, data, notes)
    console.print(table)

    choice = Prompt.ask("\nSelect config", choices=[str(i) for i in range(1, len(STRATE2_CONFIGS) + 1)] + ["q"], default="q")
    if choice == "q":
        return

    idx = int(choice) - 1
    cfg_name, cfg_path, cfg_params, data_dir, _ = STRATE2_CONFIGS[idx]
    console.print(f"\n[cyan]Selected: {cfg_name} ({cfg_params})[/]")

    # Check VM
    state = tpu.vm_state()
    if state != "READY":
        console.print(f"[red]TPU is {state}. Use [2] Start VM first.[/]")
        wait_key()
        return

    # Auto code sync
    console.print("[dim]Syncing code...[/]")
    rc, _ = tpu.scp_upload()
    if rc != 0:
        console.print("[red]Code sync failed.[/]")
        wait_key()
        return

    # Auto data check + stage
    if not _ensure_data(tpu, data_dir):
        console.print("[red]Data staging failed. Aborting.[/]")
        wait_key()
        return

    # Resume?
    resume = Confirm.ask("Resume from checkpoint?", default=False)

    # Derive scale tier from config name
    scale_tier_map = {"v6e_xs": "xs", "v6e_s": "s", "v6e_m": "m", "v6e_l": "l"}
    scale_tier = scale_tier_map.get(cfg_name, cfg_name)

    # Build remote launch command
    env_exports = (
        f'export PYTHONPATH="$HOME/Financial_IA" && '
        f'export SCALE_CONFIG="{cfg_path}" && '
        f'export SCALE_TIER="{scale_tier}" && '
        f'export RESUME="{"true" if resume else "false"}" && '
        f'export GCS_BUCKET="{tpu.cfg.gcs}" && '
        f'export TPU_TYPE="{tpu.cfg.tpu_type}" && '
        f'export TPU_GEN="v6e" && '
        f'export JAX_PLATFORMS=tpu && '
        f'export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_cache" && '
        f'mkdir -p "$HOME/.jax_cache" && '
        f'export LIBTPU_INIT_ARGS="{LIBTPU_INIT_ARGS}" && '
        f'export TF_CPP_MIN_LOG_LEVEL=2'
    )

    resume_cmd = ""
    if resume:
        resume_cmd = (
            f'CKPT_DIR="checkpoints/jax_v6e/{scale_tier}" && '
            f'mkdir -p "$CKPT_DIR" && '
            f'gsutil -m rsync -r "{tpu.cfg.gcs}/checkpoints/jax_v6e/{scale_tier}/" "$CKPT_DIR/" 2>/dev/null; '
        )

    launch_cmd = (
        f'source .venv_tpu/bin/activate && '
        f'{env_exports} && '
        f'{resume_cmd}'
        f'nohup python3 -u scripts/run_training.py > training_v6e.log 2>&1 &'
    )

    console.print(f"\n[cyan]Launching training ({cfg_name}, {cfg_params})...[/]")
    rc, out = tpu.ssh(launch_cmd, timeout=30)
    if rc != 0:
        console.print(f"[red]Launch failed:\n{out}[/]")
    else:
        console.print("[green]Training launched![/] Use [bold][5] Monitor[/] to track progress.")
    wait_key()


# ── [4] Train RL ─────────────────────────────────────────────────────────────

def screen_train_rl(tpu: TPU):
    console.print("\n[bold]Train RL — Mode Selection[/]\n")

    for key, label, script, _, _, _ in RL_MODES:
        console.print(f"  [bold green][{key}][/] {label}")
    console.print(f"  [bold green][q][/] Back")

    choice = Prompt.ask("\nSelect mode", choices=[m[0] for m in RL_MODES] + ["q"], default="q")
    if choice == "q":
        return

    mode = next(m for m in RL_MODES if m[0] == choice)
    _, label, script, buffer_dir, logfile, defaults = mode

    console.print(f"\n[cyan]Selected: {label}[/]")

    state = tpu.vm_state()
    if state != "READY":
        console.print(f"[red]TPU is {state}. Use [2] Start VM first.[/]")
        wait_key()
        return

    # Code sync
    console.print("[dim]Syncing code...[/]")
    tpu.scp_upload()

    # Check RL buffer
    rc, out = tpu.ssh(f"test -d {buffer_dir} && ls {buffer_dir}/*.npz 2>/dev/null | wc -l", timeout=15)
    has_buffer = rc == 0 and out.strip().isdigit() and int(out.strip()) > 0
    if not has_buffer:
        console.print(f"[yellow]RL buffer not found at {buffer_dir}[/]")
        console.print("[dim]Trying GCS sync...[/]")
        rc, _ = tpu.ssh(
            f'mkdir -p {buffer_dir} && gsutil -m rsync -r {tpu.cfg.gcs}/data/{buffer_dir.replace("data/", "")} {buffer_dir}/',
            timeout=300,
        )
        if rc != 0:
            console.print("[red]Buffer sync failed. Run precompute_rl_buffer.py first.[/]")
            wait_key()
            return

    # Ask hyperparams
    args_str = ""
    for param_name, default_val in defaults:
        val = Prompt.ask(f"  {param_name}", default=default_val)
        args_str += f" --{param_name} {val}"

    launch_cmd = (
        f'source .venv_tpu/bin/activate && '
        f'export PYTHONPATH="$HOME/Financial_IA" && '
        f'export JAX_PLATFORMS=tpu && '
        f'export LIBTPU_INIT_ARGS="{LIBTPU_INIT_ARGS}" && '
        f'export TF_CPP_MIN_LOG_LEVEL=2 && '
        f'nohup python3 -u {script} --buffer_dir {buffer_dir}{args_str} > {logfile} 2>&1 &'
    )

    console.print(f"\n[cyan]Launching {label}...[/]")
    rc, out = tpu.ssh(launch_cmd, timeout=30)
    if rc != 0:
        console.print(f"[red]Launch failed:\n{out}[/]")
    else:
        console.print(f"[green]Training launched![/] Log: {logfile}. Use [bold][5] Monitor[/].")
    wait_key()


# ── Braille chart renderer ───────────────────────────────────────────────────

# Braille block: 2 columns x 4 rows per character (U+2800..U+28FF)
# Dot positions:  col0: (0,0)=0x01 (1,0)=0x02 (2,0)=0x04 (3,0)=0x40
#                 col1: (0,1)=0x08 (1,1)=0x10 (2,1)=0x20 (3,1)=0x80
_BRAILLE_BASE = 0x2800
_DOT_MAP = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def _render_braille_chart(
    values: list[float],
    width: int = 60,
    height: int = 15,
    color: str = "green",
    title: str = "",
    y_label: str = "",
    steps: list[int] | None = None,
) -> str:
    """Render a list of values as a clean ASCII line chart.

    Uses half-block characters (▄▀█) for smooth curves with horizontal
    grid lines for readability. Y-axis with 5 tick marks, X-axis with
    real step numbers.

    Returns a string ready for rich console output (with color markup).
    """
    if not values or len(values) < 2:
        return "[dim]Not enough data points for a graph.[/]"

    # Subsample to fit width
    if len(values) > width:
        indices = [int(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        sampled = [values[i] for i in indices]
        step_indices = [steps[i] if steps else i for i in indices]
    else:
        sampled = list(values)
        step_indices = list(steps) if steps else list(range(len(values)))

    n_cols = len(sampled)
    n_rows = height

    v_min = min(sampled)
    v_max = max(sampled)
    margin = (v_max - v_min) * 0.05
    if v_max == v_min:
        margin = 1.0
    v_min -= margin
    v_max += margin

    # Map value to row (0=top, n_rows-1=bottom)
    def to_row(v: float) -> int:
        return max(0, min(n_rows - 1, int((v_max - v) / (v_max - v_min) * (n_rows - 1))))

    # Format Y value compactly
    def fmt_y(v: float) -> str:
        if abs(v) >= 10000:
            return f"{v/1000:>7.1f}K"
        elif abs(v) >= 100:
            return f"{v:>8.0f}"
        elif abs(v) >= 1:
            return f"{v:>8.2f}"
        else:
            return f"{v:>8.4f}"

    # Y-axis tick values (5 ticks)
    n_ticks = 5
    y_ticks = {}
    for i in range(n_ticks):
        row = int(i * (n_rows - 1) / (n_ticks - 1))
        val = v_max - (v_max - v_min) * i / (n_ticks - 1)
        y_ticks[row] = val

    # Build character grid
    lines: list[str] = []

    for r in range(n_rows):
        # Y-axis label
        if r in y_ticks:
            label = fmt_y(y_ticks[r])
        else:
            label = " " * 8

        # Build row characters
        row_chars = []
        for c in range(n_cols):
            pt_row = to_row(sampled[c])
            if pt_row == r:
                row_chars.append("●")
            elif r in y_ticks:
                row_chars.append("┄")
            else:
                row_chars.append(" ")

        # Connect adjacent points with vertical lines
        for c in range(n_cols - 1):
            r0 = to_row(sampled[c])
            r1 = to_row(sampled[c + 1])
            if min(r0, r1) < r < max(r0, r1):
                row_chars[c] = "│" if row_chars[c] in (" ", "┄") else row_chars[c]

        row_str = "".join(row_chars)
        # Color the data points/lines differently from grid
        colored = ""
        for ch in row_str:
            if ch in ("●", "│"):
                colored += f"[bold {color}]{ch}[/]"
            elif ch == "┄":
                colored += f"[dim]{ch}[/]"
            else:
                colored += ch

        lines.append(f"[dim]{label}[/] │{colored}│")

    # X-axis border
    x_border = "─" * n_cols
    lines.append(f"{'':>8} └{x_border}┘")

    # X-axis labels (step numbers)
    if step_indices:
        s0 = str(step_indices[0])
        s_mid = str(step_indices[len(step_indices) // 2])
        s_end = str(step_indices[-1])
        mid_pos = n_cols // 2 - len(s_mid) // 2
        end_pos = n_cols - len(s_end)
        x_labels = s0 + " " * (mid_pos - len(s0)) + s_mid + " " * (end_pos - mid_pos - len(s_mid)) + s_end
        lines.append(f"{'':>9} [dim]{x_labels}[/]")

    header = ""
    if title:
        header = f"[bold {color}]{title}[/]"
        if y_label:
            header += f"  [dim]({y_label})[/]"
        header += "\n"

    return header + "\n".join(lines)


# ── Log parser ───────────────────────────────────────────────────────────────

RE_STEP = re.compile(r"step[=:\s]+(\d+)", re.IGNORECASE)
RE_LOSS = re.compile(r"loss[=:\s]+([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
RE_SHARPE = re.compile(r"sharpe[=:\s]+([-+]?[\d.]+)", re.IGNORECASE)
RE_LR = re.compile(r"lr[=:\s]+([\d.]+(?:e[+-]?\d+)?)", re.IGNORECASE)
RE_EVAL = re.compile(r"EVAL|eval_loss|backtest", re.IGNORECASE)


def _parse_log_metrics(log_text: str) -> dict[str, list[tuple[int, float]]]:
    """Parse step/loss/sharpe/lr from log text.

    Returns dict of metric_name -> [(step, value), ...].
    """
    metrics: dict[str, list[tuple[int, float]]] = {
        "loss": [], "sharpe": [], "lr": [],
    }
    current_step = 0
    for line in log_text.splitlines():
        m = RE_STEP.search(line)
        if m:
            current_step = int(m.group(1))

        m = RE_LOSS.search(line)
        if m:
            try:
                metrics["loss"].append((current_step, float(m.group(1))))
            except ValueError:
                pass

        m = RE_SHARPE.search(line)
        if m:
            try:
                metrics["sharpe"].append((current_step, float(m.group(1))))
            except ValueError:
                pass

        m = RE_LR.search(line)
        if m:
            try:
                metrics["lr"].append((current_step, float(m.group(1))))
            except ValueError:
                pass

    return metrics


# ── [5] Monitor ──────────────────────────────────────────────────────────────

def _select_logfile(tpu: TPU) -> str | None:
    """List and select a log file on the TPU."""
    rc, out = tpu.ssh("ls -t *.log train_*.log training_*.log Financial_IA/logs/*.log logs/*.log 2>/dev/null | sort -ru", timeout=10)
    if rc != 0 or not out.strip():
        console.print("[yellow]No log files found.[/]")
        return None

    logs = list(dict.fromkeys(l.strip() for l in out.strip().splitlines() if l.strip()))
    console.print("\nAvailable logs:")
    for i, log in enumerate(logs, 1):
        console.print(f"  [bold green][{i}][/] {log}")

    choice = Prompt.ask("Select log", choices=[str(i) for i in range(1, len(logs) + 1)] + ["q"], default="1")
    if choice == "q":
        return None
    return logs[int(choice) - 1]


def screen_monitor(tpu: TPU):
    console.print("\n[bold]Monitor Training[/]\n")
    console.print("  [bold green][a][/] Live tail       (stream log + metrics bar)")
    console.print("  [bold green][b][/] Loss graph      (parse full log, render chart)")
    console.print("  [bold green][c][/] Multi graph     (loss + sharpe + lr)")
    console.print("  [bold green][q][/] Back")

    mode = Prompt.ask("\nSelect", choices=["a", "b", "c", "q"], default="a")
    if mode == "q":
        return

    if not tpu.is_reachable():
        console.print("[red]TPU unreachable.[/]")
        wait_key()
        return

    logfile = _select_logfile(tpu)
    if not logfile:
        wait_key()
        return

    if mode == "a":
        _monitor_live_tail(tpu, logfile)
    elif mode in ("b", "c"):
        _monitor_graph(tpu, logfile, multi=(mode == "c"))


def _monitor_live_tail(tpu: TPU, logfile: str):
    """Live tail with colorized output and metrics status bar."""
    console.print(f"\n[cyan]Tailing {logfile}[/] (Ctrl+C to stop)\n")

    last_step, last_loss, last_sharpe = "?", "?", "?"

    try:
        proc = tpu.ssh_stream(f"tail -n 30 -f {logfile}")
        while True:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.5)
                continue

            m = RE_STEP.search(line)
            if m:
                last_step = m.group(1)
            m = RE_LOSS.search(line)
            if m:
                last_loss = m.group(1)
            m = RE_SHARPE.search(line)
            if m:
                last_sharpe = m.group(1)

            text = line.rstrip()
            if RE_EVAL.search(text):
                console.print(f"[bold magenta]{text}[/]")
            elif "error" in text.lower() or "nan" in text.lower():
                console.print(f"[bold red]{text}[/]")
            elif RE_LOSS.search(text):
                console.print(f"[green]{text}[/]")
            else:
                console.print(f"[dim]{text}[/]")

    except KeyboardInterrupt:
        proc.terminate()
        console.print(f"\n[dim]Stopped. Last: step={last_step} loss={last_loss} sharpe={last_sharpe}[/]")
    wait_key()


def _monitor_graph(tpu: TPU, logfile: str, multi: bool = False):
    """Fetch full log, parse metrics, render braille charts."""
    console.print(f"\n[cyan]Fetching {logfile}...[/]")
    rc, log_text = tpu.ssh(f"cat {logfile}", timeout=30)
    if rc != 0 or not log_text.strip():
        console.print("[red]Failed to fetch log.[/]")
        wait_key()
        return

    metrics = _parse_log_metrics(log_text)

    # Terminal width for chart sizing
    term_w = shutil.get_terminal_size((80, 24)).columns
    chart_w = min(term_w - 16, 100)  # leave room for y-axis labels
    chart_h = 12

    # Loss chart (always shown)
    if metrics["loss"]:
        loss_steps = [s for s, _ in metrics["loss"]]
        loss_vals = [v for _, v in metrics["loss"]]
        chart = _render_braille_chart(
            loss_vals, width=chart_w, height=chart_h,
            color="green", title="Loss", y_label=f"{len(loss_vals)} points",
            steps=loss_steps,
        )
        console.print(Panel(chart, border_style="green", padding=(1, 2)))

        # Summary stats
        console.print(
            f"  [dim]start=[/]{loss_vals[0]:.1f}  "
            f"[dim]end=[/]{loss_vals[-1]:.1f}  "
            f"[dim]min=[/]{min(loss_vals):.1f}  "
            f"[dim]max=[/]{max(loss_vals):.1f}  "
            f"[dim]ratio=[/]{loss_vals[-1]/loss_vals[0]:.3f}x"
        )
    else:
        console.print("[yellow]No loss data found in log.[/]")

    if multi:
        # Sharpe chart
        if metrics["sharpe"]:
            sharpe_steps = [s for s, _ in metrics["sharpe"]]
            sharpe_vals = [v for _, v in metrics["sharpe"]]
            chart = _render_braille_chart(
                sharpe_vals, width=chart_w, height=chart_h,
                color="cyan", title="Sharpe", y_label=f"{len(sharpe_vals)} points",
                steps=sharpe_steps,
            )
            console.print(Panel(chart, border_style="cyan", padding=(1, 2)))
            console.print(
                f"  [dim]start=[/]{sharpe_vals[0]:.4f}  "
                f"[dim]end=[/]{sharpe_vals[-1]:.4f}  "
                f"[dim]best=[/]{max(sharpe_vals):.4f}"
            )
        else:
            console.print("[dim]No sharpe data in log.[/]")

        # LR chart
        if metrics["lr"]:
            lr_steps = [s for s, _ in metrics["lr"]]
            lr_vals = [v for _, v in metrics["lr"]]
            chart = _render_braille_chart(
                lr_vals, width=chart_w, height=chart_h,
                color="yellow", title="Learning Rate", y_label=f"{len(lr_vals)} points",
                steps=lr_steps,
            )
            console.print(Panel(chart, border_style="yellow", padding=(1, 2)))
        else:
            console.print("[dim]No LR data in log.[/]")

    wait_key()


# ── [6] Data Management ─────────────────────────────────────────────────────

def screen_data(tpu: TPU):
    console.print("\n[bold]Data Management[/]\n")
    console.print("  [bold green][a][/] GCS Status       (gsutil du)")
    console.print("  [bold green][b][/] Stage Drive->GCS (trc_data_manager)")
    console.print("  [bold green][c][/] Sync GCS->TPU    (specific dataset)")
    console.print("  [bold green][d][/] Cleanup GCS      (confirm)")
    console.print("  [bold green][e][/] Backup Ckpt      (trc_data_manager)")
    console.print("  [bold green][q][/] Back")

    choice = Prompt.ask("\nSelect", choices=["a", "b", "c", "d", "e", "q"], default="q")

    if choice == "a":
        console.print("\n[cyan]GCS bucket status...[/]")
        rc, out = _run_local(f"gsutil du -sh {tpu.cfg.gcs}/data/ {tpu.cfg.gcs}/checkpoints/ {tpu.cfg.gcs}/xla_cache/ 2>&1")
        console.print(out if out.strip() else "[dim]Bucket empty or gsutil error[/]")

    elif choice == "b":
        console.print("\n[cyan]Staging Drive -> GCS via trc_data_manager.sh...[/]")
        rc, out = _run_local("bash scripts/trc_data_manager.sh stage", timeout=600)
        console.print(out[-1000:] if out else "[dim]No output[/]")

    elif choice == "c":
        dataset = Prompt.ask("Dataset to sync", default="arrayrecord_multi")
        if tpu.is_reachable():
            console.print(f"[cyan]Syncing {dataset} from GCS to TPU...[/]")
            rc, out = tpu.ssh(
                f'mkdir -p data/{dataset} && gsutil -m rsync -r {tpu.cfg.gcs}/data/{dataset}/ data/{dataset}/',
                timeout=600,
            )
            console.print(out[-500:] if out else "[green]Done[/]")
        else:
            console.print("[red]TPU unreachable.[/]")

    elif choice == "d":
        if Confirm.ask("[red]Delete all data from GCS?[/] This removes data/, checkpoints/, xla_cache/", default=False):
            console.print("[yellow]Cleaning up GCS...[/]")
            rc, out = _run_local("bash scripts/trc_data_manager.sh cleanup --force", timeout=300)
            console.print(out[-500:] if out else "[dim]No output[/]")

    elif choice == "e":
        console.print("[cyan]Backing up latest checkpoint to Drive...[/]")
        rc, out = _run_local("bash scripts/trc_data_manager.sh backup --latest", timeout=300)
        console.print(out[-500:] if out else "[dim]No output[/]")

    wait_key()


# ── [7] Checkpoints ──────────────────────────────────────────────────────────

def screen_checkpoints(tpu: TPU):
    console.print("\n[bold]Checkpoints[/]\n")
    console.print("  [bold green][a][/] List on TPU")
    console.print("  [bold green][b][/] List on GCS")
    console.print("  [bold green][c][/] Sync TPU -> GCS")
    console.print("  [bold green][d][/] Backup to Drive")
    console.print("  [bold green][q][/] Back")

    choice = Prompt.ask("\nSelect", choices=["a", "b", "c", "d", "q"], default="q")

    if choice == "a":
        if tpu.is_reachable():
            rc, out = tpu.ssh("ls -la checkpoints/*/ 2>/dev/null || echo 'No checkpoints found'", timeout=15)
            console.print(out)
        else:
            console.print("[red]TPU unreachable.[/]")

    elif choice == "b":
        rc, out = _run_local(f"gsutil ls -l {tpu.cfg.gcs}/checkpoints/ 2>&1")
        console.print(out if out.strip() else "[dim]No GCS checkpoints[/]")

    elif choice == "c":
        if tpu.is_reachable():
            subdir = Prompt.ask("Checkpoint subdir", default="jax_v6e")
            console.print(f"[cyan]Syncing checkpoints/{subdir}/ to GCS...[/]")
            rc, out = tpu.ssh(
                f'gsutil -m rsync -r checkpoints/{subdir}/ {tpu.cfg.gcs}/checkpoints/{subdir}/',
                timeout=300,
            )
            console.print(out[-500:] if out else "[green]Done[/]")
        else:
            console.print("[red]TPU unreachable.[/]")

    elif choice == "d":
        console.print("[cyan]Backing up to Drive...[/]")
        rc, out = _run_local("bash scripts/trc_data_manager.sh backup --latest", timeout=300)
        console.print(out[-500:] if out else "[dim]No output[/]")

    wait_key()


# ── [8] Stop VM ──────────────────────────────────────────────────────────────

def screen_stop_vm(tpu: TPU):
    console.print("\n[bold red]Stop TPU VM[/]")

    state = tpu.vm_state()
    if state == "NOT_FOUND":
        console.print("[dim]TPU VM does not exist.[/]")
        wait_key()
        return

    console.print(f"VM state: [bold]{state}[/]")

    if Confirm.ask("Backup checkpoints to GCS before deleting?", default=True):
        if tpu.is_reachable():
            console.print("[cyan]Syncing checkpoints to GCS...[/]")
            tpu.ssh(
                f'gsutil -m rsync -r checkpoints/ {tpu.cfg.gcs}/checkpoints/',
                timeout=300,
            )
            console.print("[green]Backup done.[/]")

    if not Confirm.ask(
        f"[bold red]Delete TPU {tpu.cfg.name}?[/] This is irreversible.",
        default=False,
    ):
        console.print("[dim]Cancelled.[/]")
        wait_key()
        return

    console.print("[yellow]Deleting TPU...[/]")
    rc, out = tpu.gcloud(
        f"compute tpus tpu-vm delete {tpu.cfg.name} --zone={tpu.cfg.zone} --quiet",
        timeout=300,
    )
    if rc == 0:
        tpu._ip = None
        console.print("[green]TPU deleted. GCS data preserved.[/]")
    else:
        console.print(f"[red]Delete failed:\n{out}[/]")
    wait_key()


# ── [9] SSH Shell ────────────────────────────────────────────────────────────

def screen_ssh(tpu: TPU):
    ip = tpu._resolve_ip()
    if not ip:
        console.print("[red]TPU unreachable.[/]")
        wait_key()
        return

    console.print(f"[cyan]Connecting to {ip}...[/] (type 'exit' to return)\n")
    os.system(
        f"ssh -t -i {os.path.expanduser(tpu.cfg.ssh_key)} "
        f"-o StrictHostKeyChecking=no "
        f"{tpu.cfg.user}@{ip}"
    )


# ── Data staging helper ─────────────────────────────────────────────────────

def _ensure_data(tpu: TPU, data_dir: str) -> bool:
    """Check if data exists on TPU, sync from GCS if needed."""
    console.print(f"[dim]Checking data: {data_dir}...[/]")

    # Check for .arecord files
    rc, out = tpu.ssh(f"ls data/{data_dir}/*.arecord 2>/dev/null | wc -l", timeout=15)
    if rc == 0 and out.strip().isdigit() and int(out.strip()) > 0:
        n = int(out.strip())
        console.print(f"  [green]{n} shards found on TPU.[/]")
        return True

    # Try syncing from GCS
    console.print(f"  [yellow]Data not found. Syncing from GCS...[/]")
    rc, out = tpu.ssh(
        f'mkdir -p data/{data_dir} && '
        f'gsutil -m rsync -r {tpu.cfg.gcs}/data/{data_dir}/ data/{data_dir}/',
        timeout=600,
    )
    if rc != 0:
        console.print(f"  [red]Sync failed: {out[-200:]}[/]")
        return False

    # Verify
    rc, out = tpu.ssh(f"ls data/{data_dir}/*.arecord 2>/dev/null | wc -l", timeout=15)
    if rc == 0 and out.strip().isdigit() and int(out.strip()) > 0:
        console.print(f"  [green]{out.strip()} shards synced.[/]")
        return True

    console.print("  [red]No .arecord files after sync.[/]")
    return False


def _run_local(cmd: str, timeout: int = 120) -> tuple[int, str]:
    """Run a local shell command."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 1, f"Timeout ({timeout}s)"
    except Exception as e:
        return 1, str(e)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    tpu = TPU()

    while True:
        clear()
        banner()
        print_menu()

        try:
            choice = Prompt.ask("\n>", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"], default="q")
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "q":
            break
        elif choice == "1":
            screen_status(tpu)
        elif choice == "2":
            screen_start_vm(tpu)
        elif choice == "3":
            screen_train_strate2(tpu)
        elif choice == "4":
            screen_train_rl(tpu)
        elif choice == "5":
            screen_monitor(tpu)
        elif choice == "6":
            screen_data(tpu)
        elif choice == "7":
            screen_checkpoints(tpu)
        elif choice == "8":
            screen_stop_vm(tpu)
        elif choice == "9":
            screen_ssh(tpu)

    console.print("[dim]Bye.[/]")


if __name__ == "__main__":
    main()
