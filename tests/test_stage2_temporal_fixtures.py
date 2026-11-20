"""
Unit test suite verifying integrity and correctness of Stage 2 Temporal Parity Fixtures.
Ensures NPZ and JSON fixtures strictly obey all mathematical, physical, and temporal invariants.
"""

import os
import json
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
from scripts.prepare_sequences import resample_sequence
from scripts.live_inference import LiveTemporalPreprocessor


class TestStage2TemporalFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = REPO_ROOT / "tests" / "fixtures" / "android_parity"
        cls.npz_path = cls.fixture_dir / "stage2_temporal.npz"
        cls.json_path = cls.fixture_dir / "stage2_temporal.json"
        cls.scaler_path = REPO_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz"
        
        cls.assertTrue(cls.npz_path.exists(), f"Missing NPZ fixture: {cls.npz_path}")
        cls.assertTrue(cls.json_path.exists(), f"Missing JSON fixture: {cls.json_path}")
        cls.assertTrue(cls.scaler_path.exists(), f"Missing Scaler params: {cls.scaler_path}")
        
        cls.npz_data = np.load(str(cls.npz_path))
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)
            
        cls.preprocessor = LiveTemporalPreprocessor(str(cls.scaler_path))

    def test_01_fixture_metadata_and_case_list(self):
        """Verify top-level metadata and all 8 required cases exist."""
        expected_cases = [
            "case_1_short_sequence",
            "case_2_full_sequence",
            "case_3_long_sequence",
            "case_4_sparse_detections",
            "case_5_single_hand_continuous",
            "case_6_two_hand_continuous",
            "case_7_alternating_hands",
            "case_8_idle_window"
        ]
        npz_cases = self.npz_data["case_names"].tolist()
        self.assertEqual(npz_cases, expected_cases)
        self.assertEqual(list(self.json_data["cases"].keys()), expected_cases)

    def test_02_reproducibility_against_live_pipeline(self):
        """Verify re-running pipeline on raw inputs reproduces fixtures bit-for-bit."""
        for case_name in self.npz_data["case_names"]:
            raw_input = self.npz_data[f"{case_name}_raw_134"]
            expected_res = self.npz_data[f"{case_name}_resampled_64"]
            expected_out = self.npz_data[f"{case_name}_output_263"]
            
            # 1. Resample
            actual_res = resample_sequence(raw_input, target_len=64)
            np.testing.assert_array_equal(actual_res, expected_res,
                                          err_msg=f"[{case_name}] Resampling mismatch!")
            
            # 2. Live preprocessing
            actual_out = self.preprocessor.process_window(actual_res)
            np.testing.assert_array_equal(actual_out, expected_out,
                                          err_msg=f"[{case_name}] Preprocessing mismatch!")

    def test_03_json_to_npz_parity(self):
        """Verify JSON and NPZ fixtures match within float32 serializing precision."""
        for case_name in self.npz_data["case_names"]:
            npz_out = self.npz_data[f"{case_name}_output_263"]
            json_out = np.array(self.json_data["cases"][case_name]["output_263"], dtype=np.float32)
            np.testing.assert_allclose(npz_out, json_out, atol=1e-6, rtol=1e-6,
                                       err_msg=f"[{case_name}] JSON vs NPZ mismatch!")

    def test_04_zero_masking_invariants(self):
        """Verify inactive frames and components remain strictly 0.0."""
        # Case 8: Idle window must be all zeros
        idle_out = self.npz_data["case_8_idle_window_output_263"]
        self.assertTrue(np.all(idle_out == 0.0), "Idle window output is not completely 0.0!")
        
        # Case 5: Single hand right continuous -> Left and interaction must be strictly 0.0
        sh_out = self.npz_data["case_5_single_hand_continuous_output_263"]
        self.assertTrue(np.all(sh_out[:, 0:63] == 0.0), "Left Hand coords not 0.0 in single right hand case")
        self.assertTrue(np.all(sh_out[:, 126] == 0.0), "Left Hand presence not 0.0 in single right hand case")
        self.assertTrue(np.all(sh_out[:, 130:134] == 0.0), "Interaction features not 0.0 in single right hand case")
        self.assertTrue(np.all(sh_out[:, 134:197] == 0.0), "Left Hand velocity not 0.0 in single right hand case")
        self.assertTrue(np.all(sh_out[:, 260:263] == 0.0), "Interaction velocity not 0.0 in single right hand case")

    def test_05_transition_spike_prevention(self):
        """Verify boundary frame 0 and appearance/disappearance transitions have zero velocity."""
        # Frame 0 velocity in all cases must be 0.0
        for case_name in self.npz_data["case_names"]:
            out = self.npz_data[f"{case_name}_output_263"]
            self.assertTrue(np.all(out[0, 134:263] == 0.0), f"[{case_name}] Frame 0 velocity is not 0.0!")
            
        # Case 7: Alternating hands transition at frame 32
        alt_out = self.npz_data["case_7_alternating_hands_output_263"]
        # Left hand disappeared at frame 32 -> velocity must be 0.0
        self.assertTrue(np.all(alt_out[32, 134:197] == 0.0), "Left Hand velocity spike on disappearance at frame 32!")
        # Right hand appeared at frame 32 -> velocity must be 0.0 (no spike from 0-clamped coord)
        self.assertTrue(np.all(alt_out[32, 197:260] == 0.0), "Right Hand velocity spike on appearance at frame 32!")


if __name__ == "__main__":
    unittest.main()
