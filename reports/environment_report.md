# EchoGesture Environment Report

**Date:** 2026-11-19  
**Target System:** Local Workstation (EchoCam Project)

---

## 1. System Hardware

| Component | Specification | Details / Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home (Build 26300, x64) | 64-bit architecture |
| **CPU** | AMD Ryzen 7 7445HS | 6 Cores / 12 Threads |
| **System Memory (RAM)** | 15,636 MB (~15.3 GB) | Physical Memory |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU | 4 GB VRAM |
| **NVIDIA Driver** | 610.64 | Compatible with modern CUDA runtimes |
| **CUDA UMD** | 13.3 | User Mode Driver level |
| **Disk Storage** | ~40 GB free on `C:` | Monitoring required during dataset processing |

---

## 2. Python Environment

| Property | Value |
| :--- | :--- |
| **Python Version** | 3.10.11 |
| **Installation Path** | `C:\Users\haris\AppData\Local\Programs\Python\Python310` |
| **pip Status** | Available |
| **Virtual Environment** | None active (Operating on global interpreter) |

---

## 3. Installed Packages & Health Status

| Package | Version | Status | Notes |
| :--- | :--- | :---: | :--- |
| **TensorFlow** | 2.15.0 | ❌ BROKEN | Import failure due to missing `six` dependency |
| **Keras** | 2.15.0 | ❌ BROKEN | Import failure tied to TensorFlow stack |
| **OpenCV** | 4.10.0 | ✅ OK | Core computer vision libraries functional |
| **MediaPipe** | 0.10.14 | ❌ BROKEN | Landmark extraction pipeline blocked by missing `six` |
| **NumPy** | 1.26.4 | ✅ OK | Core array manipulation functional |
| **Matplotlib** | 3.8.4 | ❌ BROKEN | Plotting utility import failure (`six`) |
| **scikit-learn** | — | ⚠️ NOT INSTALLED | Required for metrics and evaluation |
| **PyYAML** | — | ⚠️ NOT INSTALLED | Required for configuration file parsing |
| **six** | — | ⚠️ NOT INSTALLED | Root cause of multiple package failures |

---

## 4. GPU & CUDA Analysis

| Check / Component | Status | Details |
| :--- | :---: | :--- |
| **NVIDIA GPU Present** | **YES** | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) |
| **nvidia-smi** | **Working** | Driver 610.64 reporting active hardware |
| **CUDA Toolkit (`nvcc`)** | **Not confirmed on PATH** | Compiler / developer toolkit not detected in system PATH |
| **cuDNN** | **Not confirmed** | Runtime acceleration libraries not verified on PATH |
| **TensorFlow GPU Support** | **Cannot verify** | Blocked due to broken TensorFlow installation |

---

## 5. Compatibility Warning

> [!WARNING]
> **Legacy TensorFlow & Environment Corruption Warning:**  
> The existing TensorFlow 2.15.0 installation is non-functional and relies on legacy `tensorflow-intel` packaging. Setting up an isolated virtual environment (`.venv`) with a modern release (**TensorFlow 2.16+**) is strongly recommended to resolve dependency breakages and guarantee stable model execution on Windows.

---

## 6. Training Configuration & Benchmarks

| Metric / Parameter | GPU Configuration (Recommended) | CPU Fallback (Viable Alternative) |
| :--- | :--- | :--- |
| **Framework Version** | TensorFlow 2.16+ | TensorFlow 2.16+ |
| **Batch Size** | 32 – 64 | 32 |
| **Precision** | Mixed Precision (`fp16`) | Full Precision (`fp32`) |
| **Estimated Training Time** | **10 – 30 minutes** | **30 – 90 minutes** |
| **Feasibility Assessment** | High throughput training; ideal for rapid experimentation. | Fully viable due to compact sequence model architecture (landmark coordinate features). |

---

## 7. Storage & Resource Estimates

| Resource Category | Estimated Footprint | Notes |
| :--- | :--- | :--- |
| **Raw Dataset** | ~2 – 5 GB | Input video archives and raw clips |
| **Extracted Landmarks** | ~50 – 200 MB | Pre-extracted coordinate files (`.npy` / `.npz`) |
| **Model Checkpoints** | ~5 – 20 MB each | Weights, model architectures, and training state |

---

## 8. Critical Issues

1. **Broken TensorFlow & Shared Dependencies:** TensorFlow 2.15.0, Keras, MediaPipe, and Matplotlib fail to import due to a missing root dependency (`six`).
2. **No Active Virtual Environment:** Packages are currently installed in the global Python environment, creating vulnerability to library version conflicts.
3. **Outdated TensorFlow Architecture:** TensorFlow 2.15 on Windows utilizes outdated `tensorflow-intel` wheels; migration to TF 2.16+ is necessary.
4. **Constrained Disk Headroom:** ~40 GB free space on `C:` requires disciplined landmark caching and intermediate file cleanup.
5. **CUDA Toolkit / cuDNN Not Confirmed on PATH:** `nvcc` and cuDNN are not confirmed on PATH, preventing direct confirmation of hardware acceleration.

---

## 9. Recommended Actions

- [ ] **1. Create an Isolated Virtual Environment:**  
  Initialize and activate a fresh `.venv` using Python 3.10 (`python -m venv .venv`).
- [ ] **2. Install Fresh TensorFlow 2.16+:**  
  Install modern TensorFlow without legacy dependency mismatches (`pip install tensorflow>=2.16.0`).
- [ ] **3. Install Project Dependencies:**  
  Install missing packages including `six`, `scikit-learn`, `pyyaml`, `opencv-python`, `mediapipe`, and `matplotlib`.
- [ ] **4. Verify CUDA / GPU Acceleration:**  
  Verify whether GPU device discovery succeeds with the updated stack (`tf.config.list_physical_devices('GPU')`).
- [ ] **5. Fallback to CPU Pipeline if Needed:**  
  If CUDA/cuDNN configuration encounters platform issues, leverage the CPU fallback (30–90 min training time), which remains well-suited for landmark coordinate models.
