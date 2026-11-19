---
name: ml-project-architect
description: >-
  Use this skill when making high-level architecture decisions for the EchoGesture sign language recognition system, including model selection, pipeline design, and system-level tradeoffs.
---

# ML Project Architect: EchoGesture

## Purpose
Guide high-level architecture decisions, model selection, data pipeline design, and performance tradeoffs for the EchoGesture sign language recognition system.

## When to Activate
- When designing or evaluating neural network architectures (e.g., 3D CNNs, Transformers, LSTM/GRU sequence models) for sign language recognition.
- When structuring data ingestion, preprocessing, skeleton extraction (MediaPipe/OpenPose), and feature engineering pipelines.
- When balancing latency, memory footprint, and recognition accuracy for real-time edge or cloud deployment.

## Technical Workflow
1. **Requirements & Scope Definition**: Define target sign vocabulary size, frame rate, input resolution, and deployment constraints (mobile vs. server latency targets).
2. **Pipeline & Feature Strategy**: Establish preprocessing standards, skeleton tracking extraction (e.g., hand/pose landmarks), and sequence padding/normalization techniques.
3. **Model Evaluation & Architecture Selection**: Design hybrid spatial-temporal architectures, comparing lightweight edge models against high-capacity sequence models.
4. **Tradeoff Analysis & Optimization**: Assess quantization, pruning, and inference runtime trade-offs to ensure real-time responsiveness (<50ms per frame).
5. **Validation & Benchmarking**: Define evaluation splits, cross-validation protocols, and key metrics (Top-1/Top-5 accuracy, inference latency, model size).

## Rules
- Always prioritize real-time inference viability and low latency when selecting model backbones for EchoGesture.
- Maintain rigorous data versioning and reproducible feature extraction pipelines across all experiments.
- Never introduce unvalidated model complexity without benchmarking latency and memory overhead against baseline architectures.
- Enforce strict separation between spatial feature extraction and temporal sequence modeling.

## Validation Checklist
- [ ] Target latency and throughput benchmarks are defined and documented for the deployment environment.
- [ ] Input preprocessing and skeleton extraction pipelines are fully reproducible and tested against edge cases.
- [ ] Model architecture trade-offs (accuracy vs. parameter count) are formally reviewed and justified.
- [ ] Cross-validation splits prevent data leakage across signer identities.
- [ ] Inference runtime profiling confirms adherence to real-time processing constraints.
- [ ] Model checkpoints include complete configuration schemas and quantization metadata.
