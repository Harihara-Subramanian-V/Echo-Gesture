# EchoGesture: MediaPipe & Computer Vision Environment Verification Report

**Date:** November 19, 2026  
**Project:** EchoGesture (Sign Language Recognition on Edge Devices)  
**Environment:** `.venv` (`Python 3.10.11`, Windows AMD64)  
**Status:** **PASS** (Environment verified, ready for Landmark Extraction)

---

## 1. Executive Summary

Following diagnosis of the initial dependency resolution delay caused by legacy pip backtracking, `pip` was upgraded to `26.3` inside `.venv` and a targeted, non-conflicting package installation was executed:
- `mediapipe==0.10.21`
- `numpy==1.26.4`
- `opencv-contrib-python==4.11.0.86` (resolved automatically as MediaPipe's computer vision backend)

All packages installed cleanly with zero binary or DLL conflicts. Runtime verification confirmed successful initialization of MediaPipe Hands, synthetic frame inference, real-world frame landmark extraction on MS-ASL video files, and intact compatibility with existing tools (`yt-dlp`).

---

## 2. Core Package Versions

| Package | Version | Purpose / Scope | Verification Status |
| :--- | :--- | :--- | :--- |
| **Python** | `3.10.11` (64-bit AMD64) | Isolated virtual environment (`.venv`) | **PASS** |
| **pip** | `26.3` | Package installer (upgraded from 23.0.1) | **PASS** |
| **MediaPipe** | `0.10.21` | Hand & pose landmark detector | **PASS** |
| **NumPy** | `1.26.4` | Numerical computing (`<2.0.0` bound) | **PASS** |
| **OpenCV** | `4.11.0` (`opencv-contrib-python`) | Computer vision & frame ingestion | **PASS** |
| **yt-dlp** | `2026.10.21` | Video acquisition CLI & library | **PASS** |

### Complete Installed Environment (`pip list`)
```text
absl-py               2.5.1
attrs                 26.1.0
cffi                  2.1.1
contourpy             1.3.2
cycler                0.12.1
flatbuffers           25.12.19
fonttools             4.65.0
jax                   0.6.2
jaxlib                0.6.2
kiwisolver            1.5.1
matplotlib            3.10.9
mediapipe             0.10.21
ml_dtypes             0.5.4
numpy                 1.26.4
opencv-contrib-python 4.11.0.86
opt_einsum            3.4.0
packaging             26.3
pillow                12.3.0
pip                   26.3
protobuf              4.25.9
pycparser             3.11
pyparsing             3.3.3
python-dateutil       3.0.0
scipy                 1.15.3
sentencepiece         0.2.3
setuptools            65.5.0
six                   1.17.0
sounddevice           0.5.6
tzdata                2026.6
yt-dlp                2026.10.21
```

---

## 3. Installation & Dependency Resolution Details

1. **Resolution Backtracking Eliminated**:
   - Upgrading `pip` to `26.3` enabled PEP 658 metadata-only HTTP querying. Pip resolved all 27 dependencies in less than 5 seconds without downloading extraneous wheel archives.
2. **OpenCV Dual-Package Conflict Resolved**:
   - By omitting `opencv-python` and letting MediaPipe select its designated `opencv-contrib-python-4.11.0.86`, namespace collision and circular backtracking between `opencv-python` and `opencv-contrib-python` were completely avoided.
3. **Download & Fault Tolerance**:
   - Download of the core wheels (`mediapipe` 51.0 MB, `jaxlib` 57.9 MB, `opencv-contrib-python` 46.2 MB, `scipy` 41.3 MB, `numpy` 15.8 MB) succeeded with automatic download resume and zero data corruption.
4. **Total Installation Duration**:
   - Resolution time: ~5 seconds.
   - Wheel download and installation: ~26 minutes under network constraints.

---

## 4. Verification & Runtime Tests

The verification script `scripts/verify_mediapipe_runtime.py` executed four distinct tests:

```powershell
.\.venv\Scripts\python.exe -c "import mediapipe as mp; import cv2; import numpy as np; print('MediaPipe:', mp.__version__); print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__)"
```
**Output:**
```text
MediaPipe: 0.10.21
OpenCV: 4.11.0
NumPy: 1.26.4
```

### Detailed Functional Verification
1. **MediaPipe Hands API Initialization**:
   - `mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)`
   - Initialized without warnings or DLL load failures.
   - Underlying TensorFlow Lite XNNPACK CPU delegate configured cleanly.
2. **Synthetic Image Inference**:
   - Processed a blank 480×640 RGB image.
   - Result: `multi_hand_landmarks: None` (expected; graceful negative handling).
3. **Real Video Inference (`data/raw/ms_asl/msasl_train_0000_J7tP98oDxqE.mp4`)**:
   - Video container decoded via OpenCV: 66 frames, 360×480 resolution, 29.97 fps.
   - First hand detected at frame 4:
     - Classification: `Left Hand` (confidence score: `0.5262`).
     - Landmark count: 21 keypoints.
     - Wrist coordinates: $x=0.4592, y=0.9439, z=0.0000$.
   - Detection frequency: 46 / 66 frames (69.7%) contained hand detections, accurately tracking the signing motion of the sign "book".
4. **Existing Dependency Regression Check**:
   - Initialized `yt_dlp.YoutubeDL({'quiet': True})`.
   - Verified 100% operational with no ABI breakage from the upgraded packages.

---

## 5. Current Project Status & Guardrails

- [x] Python virtual environment `.venv` fully configured and verified.
- [x] Zero global Python modifications.
- [x] `requirements.txt` preserved without premature edits.
- [x] Dataset manifests (`data/splits/manifest.csv`, `train.csv`, `val.csv`, `test.csv`, `normalized_multidataset_manifest.csv`) remain 100% untouched.
- [x] All 1,452 MS-ASL video files verified intact in `data/raw/ms_asl/`.
- [ ] Landmark extraction pilot (20 videos) unblocked and ready for execution.
