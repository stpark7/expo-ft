#!/usr/bin/env bash
#
# Download RoboCasa365 demo data into the repo's data/ dir (git-ignored).
#
# RoboCasa's download script has no output-dir flag — it reads the save location
# from DATASET_BASE_PATH in macros_private.py. This script sets that for you (to
# <repo>/data/robocasa365) and then downloads, so the data lands inside the
# project like the checkpoints. Run from the repo root.
#
# To download MORE / DIFFERENT data (other tasks, splits, or sources such as the
# full pretraining set or mimicgen), edit the TASKS / SPLITS / SOURCE vars below.
# See the dataset catalog + flags here:
#   https://robocasa.ai/docs/build/html/datasets/using_datasets.html
set -euo pipefail

# Repo root, resolved from this script's location (works from any cwd).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MACROS="$REPO_ROOT/client/sim/robocasa365/robocasa/macros_private.py"

# --- what to download (edit these to grab more data) ---
TASKS="PickPlaceCounterToStove"   # space-separated task names; see docs link above
SPLITS="pretrain target"          # subset of: pretrain target
SOURCE="human"                    # human | mimicgen

source "$REPO_ROOT/client/sim/.venv/bin/activate"

# Create macros_private.py if it doesn't exist yet (one-time copy of macros.py).
[ -f "$MACROS" ] || python -m robocasa.scripts.setup_macros

# Point DATASET_BASE_PATH at the repo's data/ dir (idempotent — safe to re-run).
sed -i "s|^DATASET_BASE_PATH = .*|DATASET_BASE_PATH = \"$REPO_ROOT/data/robocasa365\"|" "$MACROS"

for split in $SPLITS; do
    python -m robocasa.scripts.download_datasets \
        --split "$split" --source "$SOURCE" --tasks $TASKS
done

deactivate

# Data lands at: data/robocasa365/v1.0/{pretrain,target}/.../<task>/...
