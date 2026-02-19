#!/bin/bash
# Setup dependencies on a fresh TPU VM for JAX Fin-JEPA training.
#
# Usage (from local machine):
#   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \
#       --command="bash Financial_IA/scripts/setup_tpu_vm.sh"
#
# Or after SSH:
#   bash ~/Financial_IA/scripts/setup_tpu_vm.sh
set -euo pipefail

echo "=== Financial-IA TPU VM Setup ==="
echo "Date: $(date -u)"

cd "$HOME/Financial_IA"

# ── Phase 1: Python venv ──────────────────────────────────────────────────
echo "[1/4] Creating virtual environment..."
if [ ! -d ".venv_tpu" ]; then
    python3 -m venv .venv_tpu
fi
source .venv_tpu/bin/activate
pip install --upgrade pip

# ── Phase 2: JAX (TPU) ───────────────────────────────────────────────────
echo "[2/4] Installing JAX for TPU..."
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# ── Phase 3: Project dependencies ────────────────────────────────────────
echo "[3/4] Installing project dependencies..."
pip install \
    flax \
    optax \
    diffrax \
    orbax-checkpoint \
    grain \
    tensorflow-cpu \
    numpy \
    pyyaml \
    dacite

# ── Phase 4: Verify all imports ──────────────────────────────────────────
echo "[4/4] Verifying imports..."
python3 -c "
import sys
errors = []

checks = [
    ('jax', 'import jax; print(f\"  jax {jax.__version__} — {len(jax.devices())} {jax.devices()[0].platform} devices\")'),
    ('flax', 'import flax; print(f\"  flax {flax.__version__}\")'),
    ('optax', 'import optax; print(f\"  optax {optax.__version__}\")'),
    ('diffrax', 'import diffrax; print(f\"  diffrax {diffrax.__version__}\")'),
    ('orbax', 'import orbax.checkpoint; print(f\"  orbax-checkpoint OK\")'),
    ('grain', 'import grain.python; print(f\"  grain OK\")'),
    ('tensorflow', 'import tensorflow as tf; print(f\"  tensorflow {tf.__version__}\")'),
    ('dacite', 'import dacite; print(f\"  dacite OK\")'),
    ('yaml', 'import yaml; print(f\"  pyyaml {yaml.__version__}\")'),
    ('numpy', 'import numpy as np; print(f\"  numpy {np.__version__}\")'),
]

for name, code in checks:
    try:
        exec(code)
    except Exception as e:
        errors.append(f'{name}: {e}')
        print(f'  [FAIL] {name}: {e}')

if errors:
    print(f'\\n[FATAL] {len(errors)} missing dependencies:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)

print('\\nAll imports OK.')
"

echo "=== TPU VM Setup Complete ==="
