# EchoGesture 🤟
## AI-First Sign Language Recognition for Edge Devices
### Samsung R&D Institute India | XR Hackathon 2026 | Track 2

---

## 1. Project Objective & Scope

EchoGesture is an AI-first, edge-deployable sign language recognition system designed to process exocentric (third-person) video captured from a smartphone camera and produce high-confidence lexical sign predictions entirely on-device, with optional text-to-speech feedback.

> [!IMPORTANT]
> **Technical Scope Clarification:**  
> EchoGesture is explicitly designed as a **limited-vocabulary isolated sign recognition system** (~40 discrete sign classes). It is **not** an unrestricted, open-ended continuous sign language translator. It does not transcribe unconstrained conversational signing or model grammatical inflections.

---

## 2. Core Features

- 🎯 **Accurate Isolated Sign Recognition:** Classifies isolated gestures across an initial curated lexicon of ~40 high-frequency sign classes.
- 📱 **Zero Cloud Transmission:** 100% on-device processing. Raw camera frames, extracted features, and intermediate buffers remain strictly in local memory.
- ⚡ **Strict Edge Efficiency Targets:** Target inference latency $\le 50\text{ ms}$ and an INT8 quantized model footprint under $20\text{ MB}$.
- 🛡️ **Viewpoint & Morphological Invariance:** Body-relative coordinate normalization (shoulder-based origin and scale) mitigates variations in signer position and distance.
- 📉 **Audited Robustness:** Systematic robustness evaluation against landmark dropout, coordinate jitter, and temporal speed variation.
- 🔁 **Reproducible ML Pipeline:** Clean, deterministic training, validation, evaluation, and quantization scripts backed by structured YAML configurations.

---

## 3. System Architecture

The recognition pipeline processes visual data across decoupled stages from raw video frames to categorical predictions:

```
Video Input (Camera / File)
      │
      ▼
Frame Sampling (Uniform Temporal Sampling → 30 Frames @ 224×224 RGB)
      │
      ▼
MediaPipe Landmark Extraction (Hands: 2×21×3 + Pose: 7×3 + Confidences → ~196 raw values)
      │
      ▼
Feature Engineering (Mid-Shoulder Centering, Shoulder-Width Scaling, Temporal Velocities)
      │
      ▼
Temporal Sequence Classifier (Option A: 2-Layer GRU | Option B: 3-Stage Temporal CNN)
      │
      ▼
Inference Output (Class Prediction, Top-1 Confidence, Top-5 Distribution, Latency Metric)
```

See the detailed system specification in [reports/architecture.md](reports/architecture.md).

---

## 4. Repository Structure

```
EchoGesture/
├── .agents/
│   └── skills/                       # Project-specific Antigravity agent skills
│       ├── dataset-auditor/          # Dataset evaluation and audit procedures
│       ├── feature-engineering/      # Coordinate normalization and kinematic features
│       ├── hackathon-submission/     # Submission packaging and verification
│       ├── mediapipe-landmarks/      # Landmark extraction and tracking
│       ├── ml-evaluation/            # Metrics, confusion matrices, and reporting
│       ├── ml-project-architect/     # High-level architecture decisions
│       ├── robustness-testing/       # Noise and perturbation stress testing
│       ├── sequence-model-training/  # GRU and Temporal CNN training workflows
│       ├── tflite-optimization/      # INT8 quantization and edge benchmarking
│       └── video-preprocessing/      # OpenCV frame extraction and validation
├── configs/                          # Experiment and pipeline configuration files
│   ├── dataset.yaml                  # Lexicon, split ratios, and dataset paths
│   ├── preprocessing.yaml            # Sampling parameters and MediaPipe settings
│   └── training.yaml                 # Model architectures, optimizer, and quantization
├── data/                             # Data storage (git-ignored)
│   ├── raw/                          # Raw dataset video archives
│   ├── processed/                    # Segmented or standardized video clips
│   ├── landmarks/                    # Extracted coordinate sequences (.npy/.npz)
│   └── splits/                       # Signer-independent train/val/test manifests
├── models/                           # Model artifacts (git-ignored)
│   ├── checkpoints/                  # Saved training checkpoints
│   ├── exported/                     # SavedModel / ONNX export artifacts
│   └── quantized/                    # Final optimized INT8 TFLite models
├── src/                              # Modular Python source packages
│   ├── data/                         # Manifest generation and data loaders
│   ├── preprocessing/                # OpenCV video decoding and uniform sampling
│   ├── landmarks/                    # MediaPipe Hands and Pose extractors
│   ├── features/                     # Geometric normalization and velocity transforms
│   ├── training/                     # GRU and Temporal CNN network definitions
│   ├── evaluation/                   # Evaluation metrics, confusion matrix generators
│   ├── inference/                    # Standardized Python predict() interface
│   └── optimization/                 # TFLite INT8 quantization routines
├── scripts/                          # Executable utility and pipeline scripts
├── tests/                            # Unit and regression test suite
├── reports/                          # Audit, architecture, and evaluation reports
│   ├── environment_report.md         # Hardware and dependency inspection
│   └── architecture.md               # Complete architectural specification
├── outputs/                          # Evaluation outputs, plots, and logs
├── requirements.txt                  # Pinned Python package dependencies
├── .gitignore                        # Git exclusion rules for large data/models
└── README.md                         # Project documentation
```

---

## 5. Development Phases & Roadmap

| Phase | Description | Status |
| :---: | :--- | :---: |
| **Phase 0** | Development Environment Inspection & Hardware Audit | ✅ Complete |
| **Phase 1** | Project Architecture, Directory Structure, Configs & Skills | ✅ Complete |
| **Phase 2** | Dataset Acquisition & Signer-Independent Split Auditing | ⏳ Pending Approval |
| **Phase 3** | OpenCV Video Preprocessing & Uniform Frame Sampling | ⬜ Scheduled |
| **Phase 4** | MediaPipe 3D Landmark Extraction Pipeline | ⬜ Scheduled |
| **Phase 5** | Geometric Normalization & Kinematic Feature Engineering | ⬜ Scheduled |
| **Phase 6** | Baseline Model Development (GRU vs. Temporal CNN) | ⬜ Scheduled |
| **Phase 7** | Model Training, Validation & Rigorous Evaluation | ⬜ Scheduled |
| **Phase 8** | Controlled Robustness Testing (Jitter, Dropout, Resampling) | ⬜ Scheduled |
| **Phase 9** | TFLite INT8 Quantization & Latency Benchmarking | ⬜ Scheduled |
| **Phase 10** | Standardized Python Inference Interface (`predict()`) | ⬜ Scheduled |
| **Phase 11** | Android On-Device & Edge Integration | ⬜ Scheduled |
| **Phase 12** | Final Submission Verification & Documentation | ⬜ Scheduled |

---

## 6. Hackathon Targets vs. Baseline

| Evaluation Metric | Target Threshold | Baseline Status |
| :--- | :---: | :---: |
| **INT8 Quantized Model Size** | $< 20\text{ MB}$ | Pending Phase 9 |
| **Model Inference Latency** | $\le 50\text{ ms}$ | Pending Phase 9 |
| **Top-1 Recognition Accuracy** | $\ge 70.0\%$ | Pending Phase 7 |
| **Robustness Degradation** | $\le 5.0\text{ percentage points}$ | Pending Phase 8 |
| **Cloud Video Streaming** | **Zero (0%)** | Enforced by Architecture |

> [!NOTE]
> All targets represent design objectives. No performance claim will be stated without empirical validation and measured test logs.

---

## 7. Installation & Environment Setup

### Prerequisites
- Windows 11 / Linux x86_64
- Python 3.10.x
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/EchoCam/EchoGesture.git
cd EchoGesture

# Create an isolated virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 8. Verified Hardware Environment

Hardware profile inspected during Phase 0:
- **Processor:** AMD Ryzen 7 7445HS (6 Cores / 12 Threads)
- **Host Memory:** 15.6 GB RAM
- **Dedicated GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **NVIDIA Driver:** 610.64 (CUDA UMD 13.3)
- **Primary Training Engine:** CPU / Dedicated GPU via isolated virtual environment
- **Free Disk Headroom:** ~40 GB free on drive `C:`

Detailed findings are documented in [reports/environment_report.md](reports/environment_report.md).

---

## 9. Limitations & Explicit Caveats

1. **Vocabulary Ceiling:** The system is constrained to a fixed vocabulary (~40 lexical signs). Signs outside the selected vocabulary are not recognized.
2. **Grammar & Sentence Semantics:** EchoGesture does not parse spatial grammar, facial grammatical syntax, or continuous sentence sequences.
3. **Occlusion Sensitivity:** Extreme hand-over-hand occlusions or signers positioned outside the camera field of view may degrade landmark fidelity.
4. **Lighting Conditions:** Landmark detection requires sufficient ambient lighting to preserve hand edge contrast; low-light degradation will be systematically benchmarked.

---

## 10. License & Acknowledgments

- **Dataset Source:** Word-Level American Sign Language ([WLASL](https://dxli94.github.io/WLASL/))
- **Perception Backbone:** Google [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe)
- **Optimization Runtime:** [TensorFlow Lite](https://www.tensorflow.org/lite)
- **Host Competition:** Samsung R&D Institute India — XR Hackathon 2026 (Track 2)
