"""Preemption watcher for GCP preemptible/Spot TPU VMs.

Polls the GCP instance metadata server at ~1Hz to detect imminent preemption.
Pure Python thread — no Lightning dependency (unlike infra/spot_watcher.py).

Safe off-GCP: the metadata request fails silently, so this is a no-op on
local dev machines or non-GCP environments.

Usage:
    watcher = PreemptionWatcher()
    watcher.start()
    # In training loop:
    if watcher.should_stop():
        save_emergency_checkpoint()
        break
    # Cleanup:
    watcher.stop()
"""

import logging
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

log = logging.getLogger(__name__)

_METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/"
    "instance/preempted"
)
_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


def _poll_metadata() -> bool:
    """Check if this GCP instance is about to be preempted.

    Returns True if preemption is imminent, False otherwise.
    Silently returns False if not running on GCP.
    """
    try:
        req = Request(_METADATA_URL, headers=_METADATA_HEADERS)
        with urlopen(req, timeout=2) as resp:
            return resp.read().decode().strip().upper() == "TRUE"
    except (URLError, OSError, ValueError):
        return False


class PreemptionWatcher:
    """Background thread that polls GCP metadata for preemption signals.

    Args:
        poll_interval: Seconds between metadata polls (default: 1.0).
    """

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._preempted = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._preempted.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="preemption-watcher",
        )
        self._thread.start()
        log.info("PreemptionWatcher started (poll interval: %.1fs)", self.poll_interval)

    def stop(self):
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None

    def should_stop(self) -> bool:
        """Return True if preemption has been detected."""
        return self._preempted.is_set()

    def _poll_loop(self):
        """Background thread: poll metadata at ~1Hz."""
        while not self._stop_event.is_set():
            if _poll_metadata():
                log.warning(
                    "PREEMPTION DETECTED! (~30s remaining) "
                    "Save checkpoint immediately."
                )
                self._preempted.set()
                return
            time.sleep(self.poll_interval)
