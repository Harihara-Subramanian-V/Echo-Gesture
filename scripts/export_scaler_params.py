"""
EchoGesture Phase 7B.3: Scaler Parameters Exporter
Exports frozen training-fitted normalization parameters from
data/processed/sequences_v3/scaler_params.npz into android/app/src/main/assets/scaler_params.json.
Validates exact numerical agreement.
"""

import sys
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

def export_scaler(
    source_npz: Path = REPO_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz",
    target_json: Path = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "scaler_params.json"
):
    print("=== EchoGesture Scaler Parameters Export ===")
    print(f"Source NPZ: {source_npz}")
    print(f"Target JSON: {target_json}")
    
    assert source_npz.exists(), f"Source scaler not found: {source_npz}"
    target_json.parent.mkdir(parents=True, exist_ok=True)
    
    data = np.load(str(source_npz))
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    
    assert mean.shape == (134,), f"Expected mean shape (134,), got {mean.shape}"
    assert std.shape == (134,), f"Expected std shape (134,), got {std.shape}"
    assert not np.isnan(mean).any(), "Mean contains NaNs!"
    assert not np.isnan(std).any(), "Std contains NaNs!"
    assert np.all(std > 0.0), "Std contains non-positive values!"
    
    payload = {
        "metadata": {
            "description": "EchoGesture Phase 4 Frozen Scaler Parameters",
            "feature_dim": 134,
            "source_file": source_npz.as_posix()
        },
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std]
    }
    
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"Wrote {target_json.stat().st_size} bytes.")
    
    # Validation
    with open(target_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    loaded_mean = np.array(loaded["mean"], dtype=np.float32)
    loaded_std = np.array(loaded["std"], dtype=np.float32)
    
    np.testing.assert_array_equal(mean, loaded_mean, err_msg="Mean verification failed!")
    np.testing.assert_array_equal(std, loaded_std, err_msg="Std verification failed!")
    print("Verification PASSED: 100% exact float32 equality.")

if __name__ == "__main__":
    export_scaler()
