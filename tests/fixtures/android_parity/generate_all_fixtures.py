"""
EchoGesture Phase 7B.1: Master Reference Fixture Generator
Orchestrates the sequential generation of all three Android parity fixture stages:
- Stage 1: Feature Pipeline (134-D)
- Stage 2: Temporal Preprocessing (263-D)
- Stage 3: INT8 Inference Parity ([1, 64, 263] -> [1, 40])

Verifies:
1. Determinism across consecutive regeneration runs.
2. Bit-exact SHA-256 checksum stability.
3. Dimension and type invariants across all generated artifacts.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PARITY_DIR = PROJECT_ROOT / "tests" / "fixtures" / "android_parity"


def compute_file_sha256(file_path: Path) -> str:
    """Compute uppercase SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def run_stage1():
    """Execute Stage 1 feature pipeline generator."""
    from tests.fixtures.android_parity.generate_stage1_feature_fixtures import main as stage1_main
    print("\n" + "=" * 80)
    print("RUNNING STAGE 1: 134-D Feature Pipeline Fixture Generator")
    print("=" * 80)
    t0 = time.perf_counter()
    stage1_main()
    print(f"Stage 1 finished in {time.perf_counter() - t0:.2f}s")


def run_stage2():
    """Execute Stage 2 temporal preprocessing generator."""
    from tests.fixtures.android_parity.generate_stage2_temporal_fixtures import main as stage2_main
    print("\n" + "=" * 80)
    print("RUNNING STAGE 2: Temporal Preprocessing & Velocity Fixture Generator")
    print("=" * 80)
    t0 = time.perf_counter()
    stage2_main()
    print(f"Stage 2 finished in {time.perf_counter() - t0:.2f}s")


def run_stage3():
    """Execute Stage 3 INT8 inference generator."""
    from tests.fixtures.android_parity.generate_stage3_int8_fixtures import build_and_validate_fixtures as stage3_main
    print("\n" + "=" * 80)
    print("RUNNING STAGE 3: INT8 Inference Parity Fixture Generator")
    print("=" * 80)
    t0 = time.perf_counter()
    stage3_main()
    print(f"Stage 3 finished in {time.perf_counter() - t0:.2f}s")


def verify_determinism():
    """Regenerate fixtures and verify bit-for-bit deterministic equality."""
    print("\n" + "=" * 80)
    print("VERIFYING DETERMINISM: Snapshot Hash -> Regenerate -> Compare")
    print("=" * 80)

    fixture_files = [
        "stage1_features.npz",
        "stage2_temporal.npz",
        "stage3_inference.npz",
    ]

    hashes_run1 = {}
    for fname in fixture_files:
        fpath = PARITY_DIR / fname
        assert fpath.exists(), f"Missing fixture file: {fpath}"
        hashes_run1[fname] = compute_file_sha256(fpath)
        print(f"Run 1 SHA-256 [{fname}]: {hashes_run1[fname]} ({fpath.stat().st_size:,} bytes)")

    print("\nRe-executing all generators for determinism verification...")
    run_stage1()
    run_stage2()
    run_stage3()

    print("\nComparing Run 2 SHA-256 against Run 1...")
    all_match = True
    for fname in fixture_files:
        fpath = PARITY_DIR / fname
        hash_run2 = compute_file_sha256(fpath)
        print(f"Run 2 SHA-256 [{fname}]: {hash_run2}")
        if hash_run2 != hashes_run1[fname]:
            print(f"  [ERROR] Non-deterministic difference in {fname}!")
            all_match = False
        else:
            print(f"  [PASS] Bit-for-bit identical across runs.")

    assert all_match, "Determinism verification failed! One or more fixture files changed across runs."
    print("\nALL FIXTURES DEMONSTRATE 100% BIT-FOR-BIT DETERMINISM.")


def main():
    print("=" * 80)
    print("EchoGesture Phase 7B.1: Master Reference Fixture Generation & Validation")
    print("=" * 80)
    t_start = time.perf_counter()

    run_stage1()
    run_stage2()
    run_stage3()

    verify_determinism()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL FIXTURE STAGES GENERATED AND DETERMINISM VERIFIED IN {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
