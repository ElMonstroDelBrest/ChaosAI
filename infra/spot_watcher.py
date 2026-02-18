"""GCP Spot Preemption Watcher — Lightning Callback.

Polls the GCP instance metadata server at ~1Hz to detect preemption.
On detection (30s before SIGTERM): saves an emergency checkpoint to local
NVMe SSD, then sends SIGTERM to trigger the bash cleanup trap (GCS sync +
poweroff).

Safe off-GCP: the metadata request fails silently, making this callback
a no-op on local dev machines.

Usage:
    from infra.spot_watcher import SpotPreemptionCallback
    trainer = pl.Trainer(callbacks=[SpotPreemptionCallback()])
"""

import os
import signal
import threading
import time
from pathlib import Path

import pytorch_lightning as pl

# Use urllib to avoid hard dependency on requests
from urllib.request import Request, urlopen
from urllib.error import URLError


METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/preempted"
)
METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def _poll_metadata() -> bool:
    """Check if this GCP Spot instance is about to be preempted.

    Returns True if preemption is imminent, False otherwise.
    Silently returns False if not running on GCP.
    """
    try:
        req = Request(METADATA_URL, headers=METADATA_HEADERS)
        with urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip().upper() == "TRUE"
    except (URLError, OSError, ValueError):
        return False


class SpotPreemptionCallback(pl.Callback):
    """Polls GCP metadata for preemption, saves emergency checkpoint.

    Args:
        poll_interval: Seconds between metadata polls (default: 1.0).
        checkpoint_dir: Where to save the emergency checkpoint.
            Defaults to /mnt/disks/local-ssd/emergency_ckpt/ if NVMe is
            available, else checkpoints/strate_ii/.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        checkpoint_dir: str | None = None,
    ):
        super().__init__()
        self.poll_interval = poll_interval
        self._trainer: pl.Trainer | None = None
        self._preempted = threading.Event()
        self._thread: threading.Thread | None = None

        # Determine checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            nvme = Path("/mnt/disks/local-ssd/emergency_ckpt")
            fallback = Path("checkpoints/strate_ii")
            self.checkpoint_dir = nvme if nvme.parent.exists() else fallback

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ):
        """Start the polling thread when training begins."""
        if stage != "fit":
            return
        self._trainer = trainer
        self._preempted.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="spot-preemption-watcher",
        )
        self._thread.start()
        rank = getattr(trainer, "global_rank", 0)
        if rank == 0:
            print(
                f"[SpotWatcher] Polling metadata at {self.poll_interval}s "
                f"intervals, emergency ckpt dir: {self.checkpoint_dir}"
            )

    def _poll_loop(self):
        """Background thread: poll metadata at ~1Hz."""
        while not self._preempted.is_set():
            if _poll_metadata():
                print(
                    "\n[SpotWatcher] PREEMPTION DETECTED! "
                    "(30s remaining) Saving emergency checkpoint..."
                )
                self._preempted.set()
                self._emergency_save()
                return
            time.sleep(self.poll_interval)

    def _emergency_save(self):
        """Save checkpoint and request clean exit."""
        if self._trainer is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / "emergency_preempt.ckpt"

        try:
            self._trainer.save_checkpoint(str(ckpt_path))
            print(f"[SpotWatcher] Emergency checkpoint saved: {ckpt_path}")
        except Exception as e:
            print(f"[SpotWatcher] WARNING: checkpoint save failed: {e}")

        # SIGTERM → triggers the bash trap (GCS sync + poweroff)
        print("[SpotWatcher] Sending SIGTERM for clean shutdown...")
        os.kill(os.getpid(), signal.SIGTERM)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Safety check: stop training loop if preemption was detected."""
        if self._preempted.is_set():
            trainer.should_stop = True

    def teardown(self, trainer, pl_module, stage):
        """Clean up the polling thread."""
        self._preempted.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
