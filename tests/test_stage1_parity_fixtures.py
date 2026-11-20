"""
Unit test suite verifying Stage 1 Android Parity Fixtures.
Validates tests/fixtures/android_parity/stage1_features.npz and .json.
"""

import json
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "android_parity"


class TestStage1ParityFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npz_path = FIXTURES_DIR / "stage1_features.npz"
        cls.json_path = FIXTURES_DIR / "stage1_features.json"
        assert cls.npz_path.exists(), f"Missing {cls.npz_path}"
        assert cls.json_path.exists(), f"Missing {cls.json_path}"

        cls.npz_data = np.load(cls.npz_path)
        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.json_data = json.load(f)

    def test_01_fixture_files_exist_and_non_empty(self):
        self.assertGreater(self.npz_path.stat().st_size, 1000)
        self.assertGreater(self.json_path.stat().st_size, 10000)

    def test_02_all_keys_present_in_npz(self):
        expected_keys = [
            "case_1_left_only",
            "case_2_right_only",
            "case_3_both_hands",
            "case_4_missing_hands",
            "case_5_transition_sequence",
            "case_6_hungarian_tracking",
            "case_6_hungarian_frame0",
            "case_6_hungarian_frame1",
            "case_7_aspect_ratio_640x480",
            "case_7_aspect_ratio_1280x720",
            "case_7_aspect_ratio_1920x1080"
        ]
        for k in expected_keys:
            self.assertIn(k, self.npz_data.files, f"Key {k} missing from npz")

    def test_03_no_nans_or_infs_and_float32(self):
        for k in self.npz_data.files:
            arr = self.npz_data[k]
            self.assertEqual(arr.dtype, np.float32, f"Array {k} not float32")
            self.assertFalse(np.isnan(arr).any(), f"Array {k} contains NaN")
            self.assertFalse(np.isinf(arr).any(), f"Array {k} contains Inf")

    def test_04_dimension_checks(self):
        single_frame_keys = [
            "case_1_left_only",
            "case_2_right_only",
            "case_3_both_hands",
            "case_4_missing_hands",
            "case_6_hungarian_frame0",
            "case_6_hungarian_frame1",
            "case_7_aspect_ratio_640x480",
            "case_7_aspect_ratio_1280x720",
            "case_7_aspect_ratio_1920x1080"
        ]
        for k in single_frame_keys:
            self.assertEqual(self.npz_data[k].shape, (134,), f"Key {k} shape mismatch")

        self.assertEqual(self.npz_data["case_5_transition_sequence"].shape, (5, 134))
        self.assertEqual(self.npz_data["case_6_hungarian_tracking"].shape, (2, 134))

    def test_05_case_1_left_only_invariants(self):
        c1 = self.npz_data["case_1_left_only"]
        self.assertEqual(c1[126], 1.0)
        self.assertEqual(c1[127], 0.0)
        self.assertAlmostEqual(float(c1[128]), 0.95, places=5)
        self.assertEqual(c1[129], 0.0)
        self.assertEqual(c1[130], 0.0)
        self.assertEqual(c1[131], 0.0)
        self.assertTrue(np.all(c1[63:126] == 0.0))
        self.assertTrue(np.all(c1[132:134] == 0.0))

    def test_06_case_2_right_only_invariants(self):
        c2 = self.npz_data["case_2_right_only"]
        self.assertEqual(c2[126], 0.0)
        self.assertEqual(c2[127], 1.0)
        self.assertEqual(c2[128], 0.0)
        self.assertAlmostEqual(float(c2[129]), 0.92, places=5)
        self.assertEqual(c2[130], 0.0)
        self.assertEqual(c2[131], 0.0)
        self.assertTrue(np.all(c2[0:63] == 0.0))
        self.assertTrue(np.all(c2[132:134] == 0.0))

    def test_07_case_3_both_hands_geometry(self):
        c3 = self.npz_data["case_3_both_hands"]
        self.assertEqual(c3[126], 1.0)
        self.assertEqual(c3[127], 1.0)
        self.assertEqual(c3[130], 1.0)
        # Expected inter_dist: sqrt(256^2 + (-48)^2) / 800 ≈ 0.3255764
        expected_dist = float(np.sqrt(256.0**2 + (-48.0)**2) / 800.0)
        self.assertAlmostEqual(float(c3[131]), expected_dist, places=5)
        self.assertAlmostEqual(float(c3[132]), 0.32, places=5)
        self.assertAlmostEqual(float(c3[133]), -0.06, places=5)

    def test_08_case_4_missing_hands_zero_masked(self):
        c4 = self.npz_data["case_4_missing_hands"]
        self.assertTrue(np.all(c4 == 0.0))

    def test_09_case_5_transition_sequence(self):
        c5 = self.npz_data["case_5_transition_sequence"]
        # Frame 0: All zeros
        self.assertTrue(np.all(c5[0] == 0.0))
        # Frame 1: Left only
        self.assertEqual(c5[1, 126], 1.0)
        self.assertEqual(c5[1, 127], 0.0)
        self.assertTrue(np.all(c5[1, 63:126] == 0.0))
        # Frame 2 & 3: Both hands
        self.assertEqual(c5[2, 130], 1.0)
        self.assertEqual(c5[3, 130], 1.0)
        # Frame 4: Right only
        self.assertEqual(c5[4, 126], 0.0)
        self.assertEqual(c5[4, 127], 1.0)
        self.assertTrue(np.all(c5[4, 0:63] == 0.0))

    def test_10_case_6_hungarian_tracking(self):
        c6 = self.npz_data["case_6_hungarian_tracking"]
        # Frame 0: Slot 0 = Left (x ~ 0.25), Slot 1 = Right (x ~ 0.75)
        self.assertAlmostEqual(float(c6[0, 0]), 0.25, places=3)
        self.assertAlmostEqual(float(c6[0, 63]), 0.75, places=3)
        # Frame 1: Candidate 1 (x ~ 0.26) assigned to Slot 0, Candidate 0 (x ~ 0.74) to Slot 1
        self.assertAlmostEqual(float(c6[1, 0]), 0.26, places=3)
        self.assertAlmostEqual(float(c6[1, 63]), 0.74, places=3)

    def test_11_case_7_aspect_ratio_invariance(self):
        c7_640 = self.npz_data["case_7_aspect_ratio_640x480"]
        c7_1280 = self.npz_data["case_7_aspect_ratio_1280x720"]
        c7_1920 = self.npz_data["case_7_aspect_ratio_1920x1080"]

        self.assertAlmostEqual(float(c7_640[131]), 0.3200, places=4)
        expected_1280 = 512.0 / float(np.sqrt(1280.0**2 + 720.0**2))
        self.assertAlmostEqual(float(c7_1280[131]), expected_1280, places=5)
        # 1280x720 and 1920x1080 share 16:9 aspect ratio, so normalized geometry is identical
        self.assertAlmostEqual(float(c7_1280[131]), float(c7_1920[131]), places=5)
        self.assertAlmostEqual(float(c7_1280[132]), float(c7_1920[132]), places=5)

    def test_12_json_npz_cross_verification(self):
        # Verify JSON values correspond to NPZ arrays
        c1_json = self.json_data["cases"]["case_1_left_only"]["features_134"]
        c1_npz = self.npz_data["case_1_left_only"]
        np.testing.assert_allclose(c1_json, c1_npz, atol=1e-5)

        c3_json = self.json_data["cases"]["case_3_both_hands"]["features_134"]
        c3_npz = self.npz_data["case_3_both_hands"]
        np.testing.assert_allclose(c3_json, c3_npz, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
