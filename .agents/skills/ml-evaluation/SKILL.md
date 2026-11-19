---
name: ml-evaluation
description: >-
  Use this skill when evaluating model performance, computing metrics, generating confusion matrices, and creating evaluation reports.
---

# Machine Learning Evaluation Skill for EchoGesture

## Purpose
This skill defines the standardized procedures, metrics, and validation checks for evaluating machine learning models within the **EchoGesture** sign language recognition system. It ensures robust assessment of gesture classification accuracy, latency, robustness, and generalization across diverse signers and environmental conditions.

## When to Activate
- When evaluating trained sign language gesture recognition models (e.g., PyTorch/TensorFlow models processing skeletal landmarks or video streams).
- When computing classification metrics such as Top-1/Top-5 accuracy, precision, recall, and F1-score per sign class.
- When generating confusion matrices to identify frequently confused sign gestures.
- When creating evaluation reports for model iterations or release gating.

## Technical Workflow
1. **Dataset Preparation & Test Split**: Load the held-out test split of sign language sequences, ensuring signer-independent partitioning to prevent data leakage across train, validation, and test sets.
2. **Inference Execution & Latency Profiling**: Run batch inference across test samples while recording per-frame and per-sequence inference latency, memory consumption, and throughput.
3. **Metric Computation**: Compute global and per-class metrics, including Macro/Micro F1-scores, Top-1/Top-5 accuracy, and construct normalized confusion matrices to isolate misclassification patterns among visually similar signs.
4. **Robustness & Noise Stress Testing**: Evaluate model resilience under simulated sensor noise, missing skeletal landmarks, varying lighting conditions, and different execution speeds.
5. **Evaluation Report Generation**: Compile quantitative metrics, confusion matrix visualizations, and performance summaries into structured markdown reports and log artifacts.

## Rules
- **No Claims Without Evidence**: Never declare a model "improved" or "accurate" without inspecting generated evaluation logs, classification reports, and confusion matrix artifacts.
- **Signer Independence**: Ensure all test partitions strictly exclude training subjects to faithfully measure generalization to unseen signers.
- **Resource Profiling**: Always measure and document inference latency alongside classification accuracy to ensure real-time responsiveness on target edge or client hardware.
- **Complete Outputs**: Provide fully realized evaluation scripts and reporting configurations without placeholder comments or truncated blocks.

## Validation Checklist
- [ ] Test dataset verified to be strictly signer-independent from training and validation splits.
- [ ] Inference execution logs inspected for errors, memory leaks, and latency thresholds.
- [ ] Per-class precision, recall, and F1-scores computed and documented.
- [ ] Normalized confusion matrix generated and analyzed for high-error sign pairs.
- [ ] Stress-testing metrics logged for occlusion, noise, and frame-rate variations.
- [ ] Final evaluation summary compiled into a structured report artifact.
