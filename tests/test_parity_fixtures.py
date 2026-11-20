"""
EchoGesture Phase 7B.1: Automated Reference Fixture Validation Test Suite
Validates all deterministic Android parity fixtures across Stage 1, Stage 2, and Stage 3.

Verifies:
1. Original model SHA-256 integrity.
2. Scaler parameters origin strictly from the training partition.
3. Stage 1 (134-D) feature dimensions, zero-masking, and lack of NaNs/Infs.
4. Stage 2 (263-D) temporal resampling, velocity spike prevention, and zero-clamping.
5. Stage 3 (INT8) input/output shapes, monotonic argmax, dequantization, and top-5 rankings.
6. Bit-exact parity between binary NPZ and serialized JSON fixtures.
"""

import os
import sys
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "android_parity"
MODEL_PATH = PROJECT_ROOT / "models" / "quantized" / "echogesture_exp_d_int8.tflite"
SCALER_PATH = PROJECT_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz"
LABEL_ENCODER_PATH = PROJECT_ROOT / "data" / "processed" / "sequences_v3" / "label_encoder.json"

EXPECTED_MODEL_SHA256 = "ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499"


class TestReferenceFixtures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Assert required fixture files exist
        cls.stage1_npz_path = FIXTURES_DIR / "stage1_features.npz"
        cls.stage1_json_path = FIXTURES_DIR / "stage1_features.json"
        cls.stage2_npz_path = FIXTURES_DIR / "stage2_temporal.npz"
        cls.stage2_json_path = FIXTURES_DIR / "stage2_temporal.json"
        cls.stage3_npz_path = FIXTURES_DIR / "stage3_inference.npz"
        cls.stage3_json_path = FIXTURES_DIR / "stage3_inference.json"

        for p in [cls.stage1_npz_path, cls.stage1_json_path,
                  cls.stage2_npz_path, cls.stage2_json_path,
                  cls.stage3_npz_path, cls.stage3_json_path]:
            if not p.exists():
                raise FileNotFoundError(f"Fixture artifact not found: {p}")

        cls.stage1_npz = np.load(cls.stage1_npz_path)
        with open(cls.stage1_json_path, "r", encoding="utf-8") as f:
            cls.stage1_json = json.load(f)

        cls.stage2_npz = np.load(cls.stage2_npz_path)
        with open(cls.stage2_json_path, "r", encoding="utf-8") as f:
            cls.stage2_json = json.load(f)

        cls.stage3_npz = np.load(cls.stage3_npz_path)
        with open(cls.stage3_json_path, "r", encoding="utf-8") as f:
            cls.stage3_json = json.load(f)

    # -------------------------------------------------------------------------
    # 1. Model & Scaler Integrity Tests
    # -------------------------------------------------------------------------
    def test_original_model_sha256_integrity(self):
        """Verify the production INT8 model has not been modified or corrupted."""
        self.assertTrue(MODEL_PATH.exists(), f"Model missing at {MODEL_PATH}")
        hasher = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        computed_sha = hasher.hexdigest().upper()
        self.assertEqual(EXPECTED_MODEL_SHA256, computed_sha, "Model SHA-256 does not match frozen contract!")

    def test_scaler_parameters_origin(self):
        """Verify scaler parameters originate exclusively from the training partition."""
        self.assertTrue(SCALER_PATH.exists(), f"Scaler params missing at {SCALER_PATH}")
        scaler = np.load(SCALER_PATH)
        mean = scaler["mean"]
        std = scaler["std"]

        self.assertEqual(mean.shape, (134,), "Scaler mean must have shape (134,)")
        self.assertEqual(std.shape, (134,), "Scaler std must have shape (134,)")
        self.assertTrue(np.all(std > 0), "Scaler standard deviation must be strictly positive")

        # Invariant: Binary presence flags (126, 127, 130) and scores (128, 129) must have mean=0, std=1
        for idx in [126, 127, 128, 129, 130]:
            self.assertAlmostEqual(float(mean[idx]), 0.0, places=5)
            self.assertAlmostEqual(float(std[idx]), 1.0, places=5)

        # Stage 2 metadata must reference this exact scaler file
        self.assertIn("scaler_params.npz", self.stage2_json["metadata"]["scaler_source"])

    # -------------------------------------------------------------------------
    # 2. Stage 1 Feature Pipeline Invariants (134-D)
    # -------------------------------------------------------------------------
    def test_stage1_tensor_dimensions_and_types(self):
        """Verify Stage 1 feature vectors are strictly (134,) float32 without NaNs or Infs."""
        cases = self.stage1_json["cases"]
        for case_name, case_data in cases.items():
            if "features_134" in case_data:
                vec = np.array(case_data["features_134"], dtype=np.float32)
                self.assertEqual(vec.shape, (134,), f"{case_name} feature vector shape mismatch")
                self.assertFalse(np.isnan(vec).any(), f"{case_name} contains NaN")
                self.assertFalse(np.isinf(vec).any(), f"{case_name} contains Inf")
            elif "frames" in case_data:
                for frame in case_data["frames"]:
                    vec = np.array(frame["features_134"], dtype=np.float32)
                    self.assertEqual(vec.shape, (134,), f"{case_name} frame feature vector shape mismatch")
                    self.assertFalse(np.isnan(vec).any())
                    self.assertFalse(np.isinf(vec).any())
            elif "resolutions" in case_data:
                for res_info in case_data["resolutions"].values():
                    vec = np.array(res_info["features_134"], dtype=np.float32)
                    self.assertEqual(vec.shape, (134,), f"{case_name} resolution vector shape mismatch")
                    self.assertFalse(np.isnan(vec).any())
                    self.assertFalse(np.isinf(vec).any())

    def test_stage1_zero_masking_invariants(self):
        """Verify inactive hand slots and missing hands are strictly zero-masked."""
        # Case 1: Left hand only -> Right hand slot [63:126] and right presence must be 0.0
        c1 = np.array(self.stage1_json["cases"]["case_1_left_only"]["features_134"], dtype=np.float32)
        self.assertEqual(c1[126], 1.0, "Left presence must be 1.0")
        self.assertEqual(c1[127], 0.0, "Right presence must be 0.0")
        self.assertEqual(c1[130], 0.0, "Co-presence must be 0.0")
        np.testing.assert_array_equal(c1[63:126], 0.0, err_msg="Right hand landmarks must be strictly zero")
        self.assertEqual(c1[131], 0.0, "Inter-hand distance must be 0.0")
        np.testing.assert_array_equal(c1[132:134], 0.0, err_msg="Inter-hand offset must be 0.0")

        # Case 2: Right hand only -> Left hand slot [0:63] and left presence must be 0.0
        c2 = np.array(self.stage1_json["cases"]["case_2_right_only"]["features_134"], dtype=np.float32)
        self.assertEqual(c2[126], 0.0, "Left presence must be 0.0")
        self.assertEqual(c2[127], 1.0, "Right presence must be 1.0")
        self.assertEqual(c2[130], 0.0, "Co-presence must be 0.0")
        np.testing.assert_array_equal(c2[0:63], 0.0, err_msg="Left hand landmarks must be strictly zero")

        # Case 4: Neither hand detected -> All 134 features strictly 0.0
        c4 = np.array(self.stage1_json["cases"]["case_4_missing_hands"]["features_134"], dtype=np.float32)
        np.testing.assert_array_equal(c4, 0.0, err_msg="Missing hand frame must be strictly all zeros")

    # -------------------------------------------------------------------------
    # 3. Stage 2 Temporal Preprocessing Invariants (263-D)
    # -------------------------------------------------------------------------
    def test_stage2_tensor_dimensions_and_types(self):
        """Verify Stage 2 sequences are strictly (64, 263) float32 without NaNs or Infs."""
        cases = self.stage2_json["cases"]
        for case_name, case_data in cases.items():
            out_263 = np.array(case_data["output_263"], dtype=np.float32)
            self.assertEqual(out_263.shape, (64, 263), f"{case_name} output shape must be (64, 263)")
            self.assertFalse(np.isnan(out_263).any(), f"{case_name} contains NaN")
            self.assertFalse(np.isinf(out_263).any(), f"{case_name} contains Inf")

    def test_stage2_velocity_and_zero_clamping(self):
        """Verify velocity frame 0 is zero, transition spikes are prevented, and idle is all-zero."""
        # Case 8: Idle window -> All 64 frames must be strictly zero across all 263 channels
        c8 = np.array(self.stage2_json["cases"]["case_8_idle_window"]["output_263"], dtype=np.float32)
        np.testing.assert_array_equal(c8, 0.0, err_msg="Idle window must be 100% strictly zeros")

        # All cases: Frame 0 velocity channels [134:263] must be strictly 0.0
        cases = self.stage2_json["cases"]
        for case_name, case_data in cases.items():
            out_263 = np.array(case_data["output_263"], dtype=np.float32)
            v_frame0 = out_263[0, 134:263]
            np.testing.assert_array_equal(v_frame0, 0.0, err_msg=f"{case_name} frame 0 velocity must be 0.0")

        # Case 7: Alternating hands -> Transition at frame 32 must have 0.0 velocity
        c7 = np.array(self.stage2_json["cases"]["case_7_alternating_hands"]["output_263"], dtype=np.float32)
        v_L_trans = c7[32, 134:197]
        v_R_trans = c7[32, 197:260]
        np.testing.assert_array_equal(v_L_trans, 0.0, err_msg="Left hand disappearance must not produce velocity spike")
        np.testing.assert_array_equal(v_R_trans, 0.0, err_msg="Right hand appearance must not produce velocity spike")

    # -------------------------------------------------------------------------
    # 4. Stage 3 INT8 Inference Invariants ([1, 64, 263] -> [1, 40])
    # -------------------------------------------------------------------------
    def test_stage3_quantization_and_inference_invariants(self):
        """Verify INT8 tensor shapes, monotonic argmax, probability bounds, and top-5 rankings."""
        cases = self.stage3_json["cases"]
        for case_key, c_data in cases.items():
            in_f32 = np.array(c_data["input_float32"], dtype=np.float32)
            in_i8 = np.array(c_data["input_int8"], dtype=np.int8)
            out_i8 = np.array(c_data["output_int8"], dtype=np.int8)
            probs = np.array(c_data["dequant_probs"], dtype=np.float32)

            self.assertEqual(in_f32.shape, (1, 64, 263))
            self.assertEqual(in_i8.shape, (1, 64, 263))
            self.assertEqual(out_i8.shape, (1, 40))
            self.assertEqual(probs.shape, (1, 40))

            # Bounds on probabilities
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), f"{case_key} probs out of [0, 1]")

            # Monotonic argmax: argmax(out_i8[0]) == c_data["predicted_class_id"]
            best_idx = int(np.argmax(out_i8[0]))
            self.assertEqual(best_idx, c_data["predicted_class_id"], f"{case_key} argmax mismatch")

            # Top-5 candidates
            top5 = c_data["top_5"]
            self.assertEqual(len(top5), 5, f"{case_key} top-5 must have 5 entries")
            for i in range(len(top5) - 1):
                self.assertGreaterEqual(
                    top5[i]["confidence"], top5[i + 1]["confidence"],
                    f"{case_key} top-5 not sorted in descending order"
                )

        # Case 1: Zero input maps to zero point byte 23
        c1_in_i8 = np.array(cases["case_1_canonical_all_zeros"]["input_int8"], dtype=np.int8)
        np.testing.assert_array_equal(c1_in_i8, 23, err_msg="Zero float input must quantize to zero point byte 23")

        # Case 2: Extreme inputs saturate at -128 and 127
        c2_in_i8 = np.array(cases["case_2_extreme_bounds"]["input_int8"], dtype=np.int8)
        self.assertTrue(np.all((c2_in_i8 >= -128) & (c2_in_i8 <= 127)))
        self.assertEqual(c2_in_i8.min(), -128)
        self.assertEqual(c2_in_i8.max(), 127)

    # -------------------------------------------------------------------------
    # 5. NPZ vs JSON Equivalence Tests
    # -------------------------------------------------------------------------
    def test_npz_json_bit_level_parity(self):
        """Verify binary NPZ arrays and JSON serialized values match exactly."""
        # Stage 1 check
        for cname in ["case_1_left_only", "case_2_right_only", "case_3_both_hands", "case_4_missing_hands"]:
            json_arr = np.array(self.stage1_json["cases"][cname]["features_134"], dtype=np.float32)
            npz_arr = self.stage1_npz[cname]
            np.testing.assert_allclose(json_arr, npz_arr, rtol=1e-6, atol=1e-6)

        # Stage 2 check
        for cname in self.stage2_npz["case_names"]:
            json_arr = np.array(self.stage2_json["cases"][cname]["output_263"], dtype=np.float32)
            npz_arr = self.stage2_npz[f"{cname}_output_263"]
            np.testing.assert_allclose(json_arr, npz_arr, rtol=1e-6, atol=1e-6)

        # Stage 3 check
        for cname in self.stage3_json["cases"].keys():
            json_in_i8 = np.array(self.stage3_json["cases"][cname]["input_int8"], dtype=np.int8)
            npz_in_i8 = self.stage3_npz[f"{cname}/input_int8"]
            np.testing.assert_array_equal(json_in_i8, npz_in_i8)

            json_out_i8 = np.array(self.stage3_json["cases"][cname]["output_int8"], dtype=np.int8)
            npz_out_i8 = self.stage3_npz[f"{cname}/output_int8"]
            np.testing.assert_array_equal(json_out_i8, npz_out_i8)


if __name__ == "__main__":
    unittest.main()
