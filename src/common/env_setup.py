"""Environment setup helpers — e.g. thread-pinning on TPU VMs."""

import os


def setup_tpu_env():
    """Pin OpenBLAS/OMP threads to avoid explosion on 128+ core TPU VMs."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
