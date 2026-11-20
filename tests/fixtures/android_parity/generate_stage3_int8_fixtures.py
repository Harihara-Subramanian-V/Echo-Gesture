"""
EchoGesture Phase 7B.1: Stage 3 INT8 Inference Parity Reference Generator
Generates deterministic reference fixtures from the frozen INT8 model
(models/quantized/echogesture_exp_d_int8.tflite) and the production runtime contract
(scripts/echogesture_runtime.py).

Outputs:
- tests/fixtures/android_parity/stage3_inference.npz
- tests/fixtures/android_parity/stage3_inference.json
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.augment_sequences import compute_velocity_features

# Target Constants and Invariants
EXPECTED_SHA256 = "ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499"
EXPECTED_INPUT_SHAPE = (1, 64, 263)
EXPECTED_OUTPUT_SHAPE = (1, 40)
EXPECTED_INPUT_SCALE = 0.05794047564268112
EXPECTED_INPUT_ZERO_POINT = 23
EXPECTED_OUTPUT_SCALE = 0.00390625  # 1.0 / 256.0
EXPECTED_OUTPUT_ZERO_POINT = -128


def verify_model_sha256(model_path: Path) -> str:
    """Verifies that the frozen INT8 model has the exact expected SHA-256 checksum."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    hasher = hashlib.sha256()
    with open(model_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    computed_sha = hasher.hexdigest().upper()
    if computed_sha != EXPECTED_SHA256:
        raise ValueError(
            f"Model SHA-256 mismatch!\nExpected: {EXPECTED_SHA256}\nGot:      {computed_sha}"
        )
    return computed_sha


def generate_case_1_canonical_all_zeros() -> Tuple[np.ndarray, str]:
    """
    Case 1: Canonical all zeros (1, 64, 263) float32.
    Tests baseline idle logit distribution and input quantization to zero-point byte 23.
    """
    input_float32 = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32)
    desc = "Canonical all-zeros input testing baseline idle logit distribution and input quantization to zero-point byte 23"
    return input_float32, desc


def generate_case_2_extreme_bounds() -> Tuple[np.ndarray, str]:
    """
    Case 2: Extreme values beyond scale (-100.0, +100.0).
    Asserts strict saturation clamping to signed int8 bounds [-128, 127].
    """
    input_float32 = np.zeros(EXPECTED_INPUT_SHAPE, dtype=np.float32)
    # Timesteps 0..31 set to -100.0, Timesteps 32..63 set to +100.0
    input_float32[:, :32, :] = -100.0
    input_float32[:, 32:, :] = 100.0
    desc = "Extreme bounds inputs (-100.0, +100.0) asserting strict saturation clamping to -128 and 127"
    return input_float32, desc


def generate_case_3_single_hand_motion(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Case 3: Synthetic feature tensor simulating a right-hand sign trajectory.
    Left hand absent (all 0.0), Right hand moving smoothly with anatomical finger joint offsets.
    Standardized with active-frame Z-score and velocity computed with spike-prevention rule.
    """
    seq_raw = np.zeros((64, 134), dtype=np.float32)
    # Right hand presence and handedness
    seq_raw[:, 127] = 1.0   # presence_R
    seq_raw[:, 129] = 0.95  # handedness_R

    # Left hand strictly absent
    seq_raw[:, 126] = 0.0   # presence_L
    seq_raw[:, 128] = 0.0   # handedness_L
    seq_raw[:, 130] = 0.0   # both_present
    seq_raw[:, 131:134] = 0.0  # inter_hand distance and offsets

    t = np.linspace(0.0, 1.0, 64)
    for f in range(64):
        u = t[f]
        w_x = 0.55 + 0.12 * np.cos(2 * np.pi * u)
        w_y = 0.65 - 0.25 * np.sin(np.pi * u)
        w_z = -0.05 * np.sin(2 * np.pi * u)

        # 21 landmarks: wrist + 20 finger joints
        lm = np.zeros((21, 3), dtype=np.float32)
        lm[0] = [w_x, w_y, w_z]
        for k in range(1, 21):
            angle = (k / 20.0) * (np.pi / 2.0)
            r = 0.05 + 0.03 * (k % 4)
            lm[k] = [
                w_x + r * np.cos(angle),
                w_y - r * np.sin(angle),
                w_z + 0.01 * (k % 3)
            ]
        seq_raw[f, 63:126] = lm.flatten()

    # Active-frame Z-score standardization on right hand
    seq_std = seq_raw.copy()
    seq_std[:, 63:126] = (seq_raw[:, 63:126] - mean[63:126]) / (std[63:126] + 1e-6)

    # Compute spike-free velocity features (produces 263-D)
    feat_263 = compute_velocity_features(seq_std)
    input_float32 = np.expand_dims(feat_263, axis=0)

    desc = "Synthetic right-hand single motion trajectory with active standardization and spike-free velocities"
    return input_float32, desc


def generate_case_4_two_hand_motion(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Case 4: Synthetic feature tensor simulating a two-hand sign trajectory.
    Both left and right hands present and converging toward center with interaction geometry.
    Standardized and velocity-augmented.
    """
    seq_raw = np.zeros((64, 134), dtype=np.float32)
    seq_raw[:, 126] = 1.0   # presence_L
    seq_raw[:, 127] = 1.0   # presence_R
    seq_raw[:, 128] = 0.92  # handedness_L
    seq_raw[:, 129] = 0.95  # handedness_R
    seq_raw[:, 130] = 1.0   # both_present

    w_px, h_px = 640.0, 480.0
    diag = float(np.sqrt(w_px**2 + h_px**2))
    t = np.linspace(0.0, 1.0, 64)

    for f in range(64):
        u = t[f]
        # Left wrist
        lx = 0.35 + 0.10 * np.cos(2 * np.pi * u)
        ly = 0.60 - 0.15 * np.sin(np.pi * u)
        lz = 0.02 * np.sin(np.pi * u)

        # Right wrist
        rx = 0.65 - 0.10 * np.cos(2 * np.pi * u)
        ry = 0.60 - 0.15 * np.sin(np.pi * u)
        rz = -0.02 * np.sin(np.pi * u)

        # Left hand landmarks
        lm_L = np.zeros((21, 3), dtype=np.float32)
        lm_L[0] = [lx, ly, lz]
        for k in range(1, 21):
            angle = (k / 20.0) * (np.pi / 2.0)
            r = 0.04 + 0.02 * (k % 4)
            lm_L[k] = [lx - r * np.cos(angle), ly - r * np.sin(angle), lz + 0.01 * (k % 3)]
        seq_raw[f, 0:63] = lm_L.flatten()

        # Right hand landmarks
        lm_R = np.zeros((21, 3), dtype=np.float32)
        lm_R[0] = [rx, ry, rz]
        for k in range(1, 21):
            angle = (k / 20.0) * (np.pi / 2.0)
            r = 0.04 + 0.02 * (k % 4)
            lm_R[k] = [rx + r * np.cos(angle), ry - r * np.sin(angle), rz - 0.01 * (k % 3)]
        seq_raw[f, 63:126] = lm_R.flatten()

        # Two-hand interaction geometry
        dx_px = (rx - lx) * w_px
        dy_px = (ry - ly) * h_px
        seq_raw[f, 131] = float(np.sqrt(dx_px**2 + dy_px**2) / diag)
        seq_raw[f, 132] = float(dx_px / diag)
        seq_raw[f, 133] = float(dy_px / diag)

    # Active-frame Z-score standardization
    seq_std = seq_raw.copy()
    seq_std[:, 0:63] = (seq_raw[:, 0:63] - mean[0:63]) / (std[0:63] + 1e-6)
    seq_std[:, 63:126] = (seq_raw[:, 63:126] - mean[63:126]) / (std[63:126] + 1e-6)
    seq_std[:, 131:134] = (seq_raw[:, 131:134] - mean[131:134]) / (std[131:134] + 1e-6)

    # Compute spike-free velocity features
    feat_263 = compute_velocity_features(seq_std)
    input_float32 = np.expand_dims(feat_263, axis=0)

    desc = "Synthetic two-hand coordinated sign motion with interaction distance and velocity tracking"
    return input_float32, desc


def generate_case_5_real_samples(
    test_npz_path: Path,
    sample_indices: List[int]
) -> List[Tuple[np.ndarray, str, Dict[str, Any]]]:
    """
    Case 5: Genuine sequences loaded from held-out test partition (data/processed/sequences_v3/test_sequences.npz).
    Augmented with velocity features.
    """
    data = np.load(test_npz_path)
    X = data["X"]
    y = data["y"]
    v_ids = data["video_ids"]

    results = []
    for idx in sample_indices:
        seq_134 = X[idx]
        feat_263 = compute_velocity_features(seq_134)
        input_float32 = np.expand_dims(feat_263, axis=0)
        v_id = str(v_ids[idx])
        label_id = int(y[idx])
        desc = f"Genuine held-out test clip index {idx} ({v_id}) ground truth class id {label_id}"
        meta = {
            "test_sample_index": idx,
            "video_id": v_id,
            "ground_truth_id": label_id
        }
        results.append((input_float32, desc, meta))
    return results


def run_and_record_case(
    runtime: EchoGestureRuntime,
    case_name: str,
    input_float32: np.ndarray,
    description: str,
    extra_meta: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute EchoGestureRuntime on input tensor and capture full deterministic state.
    """
    assert input_float32.shape == EXPECTED_INPUT_SHAPE, (
        f"Input shape {input_float32.shape} != {EXPECTED_INPUT_SHAPE}"
    )
    assert input_float32.dtype == np.float32, f"Input dtype {input_float32.dtype} != float32"
    assert not np.isnan(input_float32).any(), "Input contains NaN"
    assert not np.isinf(input_float32).any(), "Input contains Inf"

    # Exact affine input quantization: q_in = clip(round(x / S_in) + Z_in, -128, 127)
    input_int8 = np.clip(
        np.round(input_float32 / runtime.in_scale) + runtime.in_zero_point,
        -128, 127
    ).astype(np.int8)

    # Run inference through production runtime
    res = runtime.predict(input_float32)

    # Capture output tensor directly from interpreter
    output_int8 = runtime.interpreter.get_tensor(runtime.out_index)  # shape (1, 40) int8
    assert output_int8.shape == EXPECTED_OUTPUT_SHAPE, (
        f"Output shape {output_int8.shape} != {EXPECTED_OUTPUT_SHAPE}"
    )
    assert output_int8.dtype == np.int8, f"Output dtype {output_int8.dtype} != int8"

    # Exact output dequantization: p = (q_out - Z_out) * S_out
    dequant_probs = (
        output_int8.astype(np.float32) - float(runtime.out_zero_point)
    ) * float(runtime.out_scale)
    assert dequant_probs.shape == EXPECTED_OUTPUT_SHAPE, (
        f"Dequant probs shape {dequant_probs.shape} != {EXPECTED_OUTPUT_SHAPE}"
    )
    assert not np.isnan(dequant_probs).any(), "Dequant probs contain NaN"
    assert not np.isinf(dequant_probs).any(), "Dequant probs contain Inf"

    # Monotonic argmax invariant check
    pred_class_id = int(np.argmax(output_int8[0]))
    prob_argmax = int(np.argmax(dequant_probs[0]))
    assert pred_class_id == prob_argmax == res["predicted_class_id"], (
        f"Argmax mismatch: q_out={pred_class_id}, p={prob_argmax}, runtime={res['predicted_class_id']}"
    )

    pred_class_name = runtime.idx2label[pred_class_id]
    assert pred_class_name == res["predicted_class"], (
        f"Class name mismatch: {pred_class_name} vs {res['predicted_class']}"
    )

    conf = float(np.clip(dequant_probs[0, pred_class_id], 0.0, 1.0))
    assert abs(conf - res["confidence"]) < 1e-6, (
        f"Confidence mismatch: {conf} vs {res['confidence']}"
    )

    top_5 = res["top_5"]
    assert len(top_5) == 5, f"Expected 5 candidates, got {len(top_5)}"

    case_record = {
        "case_name": case_name,
        "description": description,
        "input_float32": input_float32,
        "input_int8": input_int8,
        "output_int8": output_int8,
        "dequant_probs": dequant_probs,
        "predicted_class_id": pred_class_id,
        "predicted_class_name": pred_class_name,
        "confidence": conf,
        "top_5": top_5
    }
    if extra_meta:
        case_record.update(extra_meta)
    return case_record


def build_and_validate_fixtures():
    """Main pipeline execution for fixture generation and verification."""
    print("=" * 78)
    print("EchoGesture Phase 7B.1: Stage 3 INT8 Inference Parity Reference Generator")
    print("=" * 78)

    model_path = REPO_ROOT / "models" / "quantized" / "echogesture_exp_d_int8.tflite"
    label_encoder_path = REPO_ROOT / "data" / "processed" / "sequences_v3" / "label_encoder.json"
    scaler_path = REPO_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz"
    test_npz_path = REPO_ROOT / "data" / "processed" / "sequences_v3" / "test_sequences.npz"

    fixtures_dir = REPO_ROOT / "tests" / "fixtures" / "android_parity"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    npz_out = fixtures_dir / "stage3_inference.npz"
    json_out = fixtures_dir / "stage3_inference.json"

    # 1. Verify Model SHA-256
    print(f"\n[1/5] Verifying model SHA-256 for: {model_path}")
    verified_sha = verify_model_sha256(model_path)
    print(f"      Verified SHA-256: {verified_sha}")

    # 2. Initialize Production Runtime
    print(f"\n[2/5] Initializing EchoGestureRuntime...")
    runtime = EchoGestureRuntime(
        model_path=str(model_path),
        label_encoder_path=str(label_encoder_path)
    )
    print(f"      Input details:  shape={runtime.expected_shape}, scale={runtime.in_scale}, zero_point={runtime.in_zero_point}")
    print(f"      Output details: scale={runtime.out_scale}, zero_point={runtime.out_zero_point}")
    assert runtime.expected_shape == EXPECTED_INPUT_SHAPE
    assert abs(runtime.in_scale - EXPECTED_INPUT_SCALE) < 1e-9
    assert runtime.in_zero_point == EXPECTED_INPUT_ZERO_POINT
    assert abs(runtime.out_scale - EXPECTED_OUTPUT_SCALE) < 1e-9
    assert runtime.out_zero_point == EXPECTED_OUTPUT_ZERO_POINT

    # Load scaler params for synthetic cases
    scaler = np.load(scaler_path)
    scaler_mean = scaler["mean"].astype(np.float32)
    scaler_std = scaler["std"].astype(np.float32)

    # 3. Generate Test Cases
    print(f"\n[3/5] Generating test cases...")
    all_cases: Dict[str, Dict[str, Any]] = {}

    # Case 1: Canonical All Zeros
    print("      - Generating Case 1: canonical_all_zeros...")
    x1, desc1 = generate_case_1_canonical_all_zeros()
    rec1 = run_and_record_case(runtime, "case_1_canonical_all_zeros", x1, desc1)
    # Check that all elements of input_int8 are exactly 23
    assert np.all(rec1["input_int8"] == 23), "Case 1 input_int8 must be all 23 (zero-point)!"
    all_cases["case_1_canonical_all_zeros"] = rec1
    print(f"        Pred: {rec1['predicted_class_name']} (id {rec1['predicted_class_id']}), Conf: {rec1['confidence']:.4f}")

    # Case 2: Extreme Bounds
    print("      - Generating Case 2: extreme_bounds...")
    x2, desc2 = generate_case_2_extreme_bounds()
    rec2 = run_and_record_case(runtime, "case_2_extreme_bounds", x2, desc2)
    # Check strict saturation clamping
    assert np.all(np.isin(rec2["input_int8"], [-128, 127])), "Case 2 input_int8 must be strictly saturated to -128 and 127!"
    all_cases["case_2_extreme_bounds"] = rec2
    print(f"        Pred: {rec2['predicted_class_name']} (id {rec2['predicted_class_id']}), Conf: {rec2['confidence']:.4f}")

    # Case 3: Single Hand Motion
    print("      - Generating Case 3: single_hand_motion...")
    x3, desc3 = generate_case_3_single_hand_motion(scaler_mean, scaler_std)
    rec3 = run_and_record_case(runtime, "case_3_single_hand_motion", x3, desc3)
    all_cases["case_3_single_hand_motion"] = rec3
    print(f"        Pred: {rec3['predicted_class_name']} (id {rec3['predicted_class_id']}), Conf: {rec3['confidence']:.4f}")

    # Case 4: Two Hand Motion
    print("      - Generating Case 4: two_hand_motion...")
    x4, desc4 = generate_case_4_two_hand_motion(scaler_mean, scaler_std)
    rec4 = run_and_record_case(runtime, "case_4_two_hand_motion", x4, desc4)
    all_cases["case_4_two_hand_motion"] = rec4
    print(f"        Pred: {rec4['predicted_class_name']} (id {rec4['predicted_class_id']}), Conf: {rec4['confidence']:.4f}")

    # Case 5: Real Test Samples (at least 3 genuine sequences)
    print("      - Generating Case 5: genuine test samples (3 clips from test_sequences.npz)...")
    # Sample indices: 1 (book), 2 (clothes), 3 (chair)
    real_sample_indices = [1, 2, 3]
    real_samples = generate_case_5_real_samples(test_npz_path, real_sample_indices)

    for i, (x_real, desc_real, meta_real) in enumerate(real_samples, start=1):
        c_name = f"case_5_real_test_sample_{i}"
        gt_name = runtime.idx2label[meta_real["ground_truth_id"]]
        meta_real["ground_truth_name"] = gt_name
        rec_real = run_and_record_case(runtime, c_name, x_real, desc_real, extra_meta=meta_real)
        all_cases[c_name] = rec_real
        print(f"        Sample {i} (index {meta_real['test_sample_index']}, GT: {gt_name}): Pred: {rec_real['predicted_class_name']} (id {rec_real['predicted_class_id']}), Conf: {rec_real['confidence']:.4f}")

    # Canonical alias: case_5_real_test_sample maps to case_5_real_test_sample_1
    all_cases["case_5_real_test_sample"] = all_cases["case_5_real_test_sample_1"].copy()
    all_cases["case_5_real_test_sample"]["case_name"] = "case_5_real_test_sample"

    # 4. Serialize to NPZ and JSON
    print(f"\n[4/5] Serializing reference fixtures to NPZ and JSON...")

    # A. Prepare NPZ dictionary
    npz_data = {
        "model_sha256": np.array(verified_sha),
        "input_scale": np.array(runtime.in_scale, dtype=np.float32),
        "input_zero_point": np.array(runtime.in_zero_point, dtype=np.int32),
        "output_scale": np.array(runtime.out_scale, dtype=np.float32),
        "output_zero_point": np.array(runtime.out_zero_point, dtype=np.int32),
    }

    for case_key, c_data in all_cases.items():
        # Store using both slash and underscore notation for universal compatibility
        # Shapes: input_float32: (1, 64, 263), input_int8: (1, 64, 263), output_int8: (1, 40), dequant_probs: (1, 40)
        npz_data[f"{case_key}/input_float32"] = c_data["input_float32"]
        npz_data[f"{case_key}/input_int8"] = c_data["input_int8"]
        npz_data[f"{case_key}/output_int8"] = c_data["output_int8"]
        npz_data[f"{case_key}/dequant_probs"] = c_data["dequant_probs"]
        npz_data[f"{case_key}/predicted_class_id"] = np.int32(c_data["predicted_class_id"])
        npz_data[f"{case_key}/confidence"] = np.float32(c_data["confidence"])
        npz_data[f"{case_key}/predicted_class_name"] = np.array(c_data["predicted_class_name"])

        # Flat underscore equivalents
        npz_data[f"{case_key}_input_float32"] = c_data["input_float32"]
        npz_data[f"{case_key}_input_int8"] = c_data["input_int8"]
        npz_data[f"{case_key}_output_int8"] = c_data["output_int8"]
        npz_data[f"{case_key}_dequant_probs"] = c_data["dequant_probs"]
        npz_data[f"{case_key}_predicted_class_id"] = np.int32(c_data["predicted_class_id"])
        npz_data[f"{case_key}_confidence"] = np.float32(c_data["confidence"])
        npz_data[f"{case_key}_predicted_class_name"] = np.array(c_data["predicted_class_name"])

    np.savez_compressed(npz_out, **npz_data)
    print(f"      Saved NPZ fixture:  {npz_out} ({npz_out.stat().st_size / 1024:.2f} KB)")

    # B. Prepare JSON dictionary
    json_cases = {}
    for case_key, c_data in all_cases.items():
        case_dict = {
            "case_name": c_data["case_name"],
            "description": c_data["description"],
            "input_float32": c_data["input_float32"].tolist(),  # shape [1, 64, 263]
            "input_int8": c_data["input_int8"].tolist(),        # shape [1, 64, 263]
            "output_int8": c_data["output_int8"].tolist(),      # shape [1, 40]
            "output_int8_flat": c_data["output_int8"][0].tolist(),  # shape [40] for flat byte consumers
            "dequant_probs": c_data["dequant_probs"].tolist(),  # shape [1, 40]
            "dequant_probs_flat": c_data["dequant_probs"][0].tolist(),  # shape [40]
            "predicted_class_id": int(c_data["predicted_class_id"]),
            "predicted_class_name": str(c_data["predicted_class_name"]),
            "confidence": float(c_data["confidence"]),
            "top_5": c_data["top_5"]
        }
        for extra_key in ["video_id", "test_sample_index", "ground_truth_id", "ground_truth_name"]:
            if extra_key in c_data:
                case_dict[extra_key] = c_data[extra_key]
        json_cases[case_key] = case_dict

    full_json = {
        "metadata": {
            "generator": "tests/fixtures/android_parity/generate_stage3_int8_fixtures.py",
            "model_path": "models/quantized/echogesture_exp_d_int8.tflite",
            "model_sha256": verified_sha,
            "input_tensor": {
                "name": "serving_default_landmark_sequence:0",
                "shape": list(EXPECTED_INPUT_SHAPE),
                "dtype": "int8",
                "scale": float(runtime.in_scale),
                "zero_point": int(runtime.in_zero_point)
            },
            "output_tensor": {
                "name": "StatefulPartitionedCall_1:0",
                "shape": list(EXPECTED_OUTPUT_SHAPE),
                "dtype": "int8",
                "scale": float(runtime.out_scale),
                "zero_point": int(runtime.out_zero_point)
            },
            "vocabulary_size": runtime.num_classes,
            "case_keys": list(all_cases.keys())
        },
        "cases": json_cases,
        # Also expose cases directly at root for convenience
        **json_cases
    }

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)
    print(f"      Saved JSON fixture: {json_out} ({json_out.stat().st_size / 1024:.2f} KB)")

    # 5. Verification Readback
    print(f"\n[5/5] Performing verification readback on saved fixtures...")
    # Verify NPZ
    loaded_npz = np.load(npz_out)
    assert str(loaded_npz["model_sha256"]) == verified_sha
    for case_key in all_cases.keys():
        assert f"{case_key}/input_float32" in loaded_npz
        assert f"{case_key}_input_float32" in loaded_npz
        in_flt = loaded_npz[f"{case_key}/input_float32"]
        in_i8 = loaded_npz[f"{case_key}/input_int8"]
        out_i8 = loaded_npz[f"{case_key}/output_int8"]
        probs = loaded_npz[f"{case_key}/dequant_probs"]
        p_id = int(loaded_npz[f"{case_key}/predicted_class_id"])
        conf = float(loaded_npz[f"{case_key}/confidence"])

        assert in_flt.shape == EXPECTED_INPUT_SHAPE
        assert in_i8.shape == EXPECTED_INPUT_SHAPE
        assert out_i8.shape == EXPECTED_OUTPUT_SHAPE
        assert probs.shape == EXPECTED_OUTPUT_SHAPE
        assert in_flt.dtype == np.float32
        assert in_i8.dtype == np.int8
        assert out_i8.dtype == np.int8
        assert probs.dtype == np.float32

        assert not np.isnan(in_flt).any()
        assert not np.isnan(probs).any()
        assert np.argmax(out_i8[0]) == p_id
        assert abs(conf - probs[0, p_id]) < 1e-6
    print("      NPZ readback verification: PASSED (All shapes, dtypes, values verified)")

    # Verify JSON
    with open(json_out, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["metadata"]["model_sha256"] == verified_sha
    for case_key in all_cases.keys():
        c_item = loaded_json["cases"][case_key]
        in_arr = np.array(c_item["input_float32"], dtype=np.float32)
        in_i8_arr = np.array(c_item["input_int8"], dtype=np.int8)
        out_i8_arr = np.array(c_item["output_int8"], dtype=np.int8)
        probs_arr = np.array(c_item["dequant_probs"], dtype=np.float32)

        assert in_arr.shape == EXPECTED_INPUT_SHAPE
        assert in_i8_arr.shape == EXPECTED_INPUT_SHAPE
        assert out_i8_arr.shape == EXPECTED_OUTPUT_SHAPE
        assert probs_arr.shape == EXPECTED_OUTPUT_SHAPE
        assert not np.isnan(in_arr).any()
        assert not np.isnan(probs_arr).any()
        assert c_item["predicted_class_id"] == int(np.argmax(out_i8_arr[0]))
        assert len(c_item["top_5"]) == 5
    print("      JSON readback verification: PASSED (All structures and arrays verified)")

    print("\n" + "=" * 78)
    print("STAGE 3 INT8 INFERENCE REFERENCE FIXTURES GENERATED & VERIFIED SUCCESSFULLY!")
    print("=" * 78)


if __name__ == "__main__":
    build_and_validate_fixtures()
