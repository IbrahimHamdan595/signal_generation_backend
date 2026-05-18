"""One-shot migration: flatten `checkpoints/*` → `checkpoints/equities/*`.

The pre-FX layout kept every artifact at the root of `checkpoints/`. With
the multi-model registry (Phase 4 of the FX-pipeline plan) we namespace
checkpoints by asset class. This script moves the existing equities
files into `checkpoints/equities/` and creates an empty `checkpoints/fx/`
ready for the upcoming FX training run.

Idempotent: re-running after a partial move only relocates files that
are still at the old location. The `dataset_cache/` directory stays
where it is (per-ticker, not asset-class specific) and is never touched.

Usage:
    cd backend && .venv/Scripts/python scripts/migrate_checkpoints.py
"""

from __future__ import annotations

import os
import shutil
import sys


ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
EQUITIES_DIR = os.path.join(ROOT, "equities")
FX_DIR       = os.path.join(ROOT, "fx")

# Single-file artifacts to relocate (any that exist get moved).
SINGLE_FILES = (
    "best_model.pt",
    "model_config.json",
    "scaler_params.json",
    "eval_report.json",
    "last_train_result.json",
    "versions.json",
    "recent_signal_probs.json",
    "drift_baseline.json",
    "walkforward_result.json",
)

# Prefix patterns whose matching files all move together.
PREFIX_PATTERNS = ("model_",)   # versioned checkpoints e.g. model_20260514_122431.pt


def _move_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        print(f"  [skip] {os.path.basename(src)} already at destination")
        return False
    shutil.move(src, dst)
    print(f"  [move] {os.path.basename(src)}")
    return True


def main() -> int:
    if not os.path.isdir(ROOT):
        print(f"[error] {ROOT} does not exist - nothing to migrate")
        return 1

    os.makedirs(EQUITIES_DIR, exist_ok=True)
    os.makedirs(FX_DIR,       exist_ok=True)
    print(f"[ok] Ensured subfolders exist: {EQUITIES_DIR}, {FX_DIR}")

    moved = 0

    # 1. Named single-file artifacts
    for fname in SINGLE_FILES:
        if _move_if_exists(os.path.join(ROOT, fname), os.path.join(EQUITIES_DIR, fname)):
            moved += 1

    # 2. Versioned checkpoint files matching the prefixes
    for entry in os.listdir(ROOT):
        full = os.path.join(ROOT, entry)
        if not os.path.isfile(full):
            continue
        if any(entry.startswith(pref) for pref in PREFIX_PATTERNS):
            if _move_if_exists(full, os.path.join(EQUITIES_DIR, entry)):
                moved += 1

    print(f"\n[done] Moved {moved} file(s) into {EQUITIES_DIR}/")
    print(f"       dataset_cache/ left untouched (not asset-class specific)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
