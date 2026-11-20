"""
EchoGesture Phase 7B.1: Stage 3 INT8 Inference Parity Fixture Invariant Tests
Validates the generated NPZ and JSON reference fixtures against mathematical invariants,
tensor dimensions, quantization consistency, and model execution parity.
"""

import os
import json
import hashlib
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestStage3Int8Fixtures(unittest.TestCase):
    """
    Unit test suite asserting the integrity of Stage 3 INT8 reference fixtures.
    """
    @classmethod
    def setUpClass(cls):
        cls.fixtures_dir = REPO_ROOT / "tests" / "fixtures" / "android_parity"
        cls.npz_path = cls.fixtures_dir / "stage3_inference.npz"
        cls.json_path = cls.fixtures_dir / "stage3_inference.json"
        cls.model_path = REPO_ROOT / "models" / "quantized" / "echogesture_exp_d_int8.tflite"

        cls.expected_sha256 = "ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499"
        cls.expected_input_shape = (1, 64, 263)
        cls.expected_output_shape = (1, 40)
        cls.expected_input_scale = 0.05794047564268112
        cls.expected_input_zero_point = 23
        cls.expected_output_scale = 0.00390625
        cls.expected_output_zero_point = -128

        # Ensure fixture files exist
        cls.assertTrue(cls, cls.npz_path.exists(), f"Missing NPZ fixture: {cls.npz_path}")
        cls.assertTrue(cls, cls.json_path.exists(), f"Missing JSON fixture: {cls.json_path}")
        cls.assertTrue(cls, cls.model_path.exists(), f"Missing model file: {cls.model_path}")

        cls.npz_data = np.load(cls.npz_path)
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)

    def test_01_model_sha256_integrity(self):
        """Verify model SHA-256 matches frozen reference contract."""
        hasher = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        computed_sha = hasher.hexdigest().upper()
        self.assertEqual(self.expected_sha256, computed_sha)
        self.assertEqual(str(self.npz_data["model_sha256"]), self.expected_sha256)
        self.assertEqual(self.json_data["metadata"]["model_sha256"], self.expected_sha256)

    def test_02_quantization_parameters(self):
        """Verify quantization scale and zero point match contract."""
        np.testing.assert_almost_equal(float(self.npz_data["input_scale"]), self.expected_input_scale, decimal=7)
        self.assertEqual(int(self.npz_data["input_zero_point"]), self.expected_input_zero_point)
        np.testing.assert_almost_equal(float(self.npz_data["output_scale"]), self.expected_output_scale, decimal=7)
        self.assertEqual(int(self.npz_data["output_zero_point"]), self.expected_output_zero_point)

    def test_03_case_coverage(self):
        """Verify all required test cases are present in both NPZ and JSON."""
        required_cases = [
            "case_1_canonical_all_zeros",
            "case_2_extreme_bounds",
            "case_3_single_hand_motion",
            "case_4_two_hand_motion",
            "case_5_real_test_sample",
            "case_5_real_test_sample_1",
            "case_5_real_test_sample_2",
            "case_5_real_test_sample_3"
        ]
        json_cases = self.json_data["cases"]
        for case in required_cases:
            self.assertIn(case, json_cases, f"Case {case} missing from JSON fixture")
            self.assertIn(f"{case}/input_float32", self.npz_data, f"{case}/input_float32 missing from NPZ")
            self.assertIn(f"{case}_input_float32", self.npz_data, f"{case}_input_float32 missing from NPZ")

    def test_04_tensor_dimensions_and_dtypes(self):
        """Verify tensor shapes, dtypes, and NaN/Inf freedom across all cases."""
        for case_name, case_dict in self.json_data["cases"].items():
            # Check NPZ arrays
            in_flt = self.npz_data[f"{case_name}/input_float32"]
            in_i8 = self.npz_data[f"{case_name}/input_int8"]
            out_i8 = self.npz_data[f"{case_name}/output_int8"]
            probs = self.npz_data[f"{case_name}/dequant_probs"]

            self.assertEqual(in_flt.shape, self.expected_input_shape)
            self.assertEqual(in_i8.shape, self.expected_input_shape)
            self.assertEqual(out_i8.shape, self.expected_output_shape)
            self.assertEqual(probs.shape, self.expected_output_shape)

            self.assertEqual(in_flt.dtype, np.float32)
            self.assertEqual(in_i8.dtype, np.int8)
            self.assertEqual(out_i8.dtype, np.int8)
            self.assertEqual(probs.dtype, np.float32)

            self.assertFalse(np.isnan(in_flt).any())
            self.assertFalse(np.isinf(in_flt).any())
            self.assertFalse(np.isnan(probs).any())
            self.assertFalse(np.isinf(probs).any())

            # Check JSON arrays
            json_in_flt = np.array(case_dict["input_float32"], dtype=np.float32)
            json_in_i8 = np.array(case_dict["input_int8"], dtype=np.int8)
            json_out_i8 = np.array(case_dict["output_int8"], dtype=np.int8)
            json_probs = np.array(case_dict["dequant_probs"], dtype=np.float32)

            self.assertEqual(json_in_flt.shape, self.expected_input_shape)
            self.assertEqual(json_in_i8.shape, self.expected_input_shape)
            self.assertEqual(json_out_i8.shape, self.expected_output_shape)
            self.assertEqual(json_probs.shape, self.expected_output_shape)

    def test_05_case_1_all_zeros_invariants(self):
        """Case 1 must quantize all float zeros to the zero point (byte 23)."""
        c1_npz_i8 = self.npz_data["case_1_canonical_all_zeros/input_int8"]
        self.assertTrue(np.all(c1_npz_i8 == 23))

        c1_json_i8 = np.array(self.json_data["cases"]["case_1_canonical_all_zeros"]["input_int8"])
        self.assertTrue(np.all(c1_json_i8 == 23))

    def test_06_case_2_extreme_bounds_saturation(self):
        """Case 2 extreme inputs must saturate strictly to -128 and 127."""
        c2_npz_i8 = self.npz_data["case_2_extreme_bounds/input_int8"]
        unique_vals = np.unique(c2_npz_i8)
        self.assertTrue(np.array_equal(unique_vals, np.array([-128, 127], dtype=np.int8)))

    def test_07_quantization_and_dequantization_math(self):
        """Verify exact mathematical alignment of affine quantization and dequantization."""
        s_in = self.expected_input_scale
        z_in = self.expected_input_zero_point
        s_out = self.expected_output_scale
        z_out = self.expected_output_zero_point

        for case_name, case_dict in self.json_data["cases"].items():
            in_flt = self.npz_data[f"{case_name}/input_float32"]
            in_i8 = self.npz_data[f"{case_name}/input_int8"]
            out_i8 = self.npz_data[f"{case_name}/output_int8"]
            probs = self.npz_data[f"{case_name}/dequant_probs"]

            # Quantization math check: q_in = clip(round(x / s_in) + z_in, -128, 127)
            expected_q_in = np.clip(np.round(in_flt / s_in) + z_in, -128, 127).astype(np.int8)
            np.testing.assert_array_equal(in_i8, expected_q_in)

            # Dequantization math check: p = (q_out - z_out) * s_out
            expected_p = (out_i8.astype(np.float32) - float(z_out)) * float(s_out)
            np.testing.assert_allclose(probs, expected_p, atol=1e-6)

            # Monotonic argmax check
            pred_id = int(self.npz_data[f"{case_name}/predicted_class_id"])
            self.assertEqual(int(np.argmax(out_i8[0])), pred_id)
            self.assertEqual(int(np.argmax(probs[0])), pred_id)
            self.assertEqual(case_dict["predicted_class_id"], pred_id)

            # Confidence check
            conf = float(self.npz_data[f"{case_name}/confidence"])
            self.assertAlmostEqual(conf, float(probs[0, pred_id]), places=6)
            self.assertAlmostEqual(case_dict["confidence"], conf, places=6)

    def test_08_top5_candidate_rankings(self):
        """Verify Top-5 rankings are properly structured, sorted, and have valid class names."""
        for case_name, case_dict in self.json_data["cases"].items():
            top_5 = case_dict["top_5"]
            self.assertEqual(len(top_5), 5)
            ranks = [item["rank"] for item in top_5]
            self.assertEqual(ranks, [1, 2, 3, 4, 5])

            confs = [item["confidence"] for item in top_5]
            for i in range(len(confs) - 1):
                self.assertGreaterEqual(confs[i], confs[i + 1])

            top1_id = top_5[0]["class_id"]
            self.assertEqual(top1_id, case_dict["predicted_class_id"])
            self.assertEqual(top_5[0]["class_name"], case_dict["predicted_class_name"])

    def test_09_real_test_samples_consistency(self):
        """Verify genuine held-out test clips have matching metadata and high prediction confidences."""
        for i in [1, 2, 3]:
            case_key = f"case_5_real_test_sample_{i}"
            c = self.json_data["cases"][case_key]
            self.assertIn("video_id", c)
            self.assertIn("ground_truth_id", c)
            self.assertIn("ground_truth_name", c)
            # High confidence (> 0.95) for all 3 selected samples
            self.assertGreater(c["confidence"], 0.95)
            self.assertEqual(c["predicted_class_id"], c["ground_truth_id"])


if __name__ == "__main__":
    unittest.main()
