---
name: feature-engineering
description: >-
  Use this skill when designing and implementing feature normalization, body-relative coordinates, velocity features, and other derived features from raw MediaPipe landmarks.
---

# Feature Engineering for EchoGesture

## Purpose
The `feature-engineering` skill governs the transformation of raw MediaPipe landmark streams into robust, invariant, and high-performance feature representations tailored for EchoGesture sign language recognition models (LSTMs, Transformers, and Temporal Convolutional Networks).

## When to Activate
Activate this skill when:
- Designing or modifying feature extraction pipelines that process raw MediaPipe Pose, Hands, or Face landmarks.
- Implementing normalization strategies (e.g., wrist-relative coordinates, shoulder-width scaling) to ensure translation, rotation, and scale invariance across different signers and camera distances.
- Computing kinematic derivatives such as velocity, acceleration, and angular displacement between consecutive sign frames.
- Preparing input tensors for sequence models in EchoGesture.

## Technical Workflow
1. **Raw Landmark Ingestion & Validation**: Load raw MediaPipe landmark frames `(x, y, z, visibility)` for wrists, hands, and upper body pose, filtering out low-confidence detections or missing frames.
2. **Anchor-Based Normalization**: Translate all landmark coordinates relative to a stable body anchor (e.g., the root wrist or midpoint between shoulders) and scale by inter-shoulder distance to achieve scale and translation invariance.
3. **Kinematic Feature Extraction**: Compute first-order (velocity) and second-order (acceleration) temporal differences across sliding time windows to capture dynamic signing motion.
4. **Tensor Assembly & Padding**: Flatten and concatenate spatial, relative, and kinematic features into standardized feature vectors per frame, and apply padding/truncation for fixed-length sequence batching.

## Rules
- Never feed raw, unnormalized screen-space pixel coordinates directly into recognition models; always apply body-relative normalization.
- Ensure temporal features (velocity/acceleration) handle boundary conditions at sequence start/end frames without introducing NaN or infinite values.
- Maintain strict dimensionality consistency across training, validation, and live inference pipelines.

## Validation Checklist
- [ ] Raw MediaPipe landmarks are correctly loaded and checked for missing frames or visibility thresholds.
- [ ] Translation invariance is verified by shifting signer position in the frame without altering output normalized coordinates.
- [ ] Scale invariance is confirmed across varying distances from the camera using shoulder-width scaling.
- [ ] Kinematic velocity and acceleration features are computed accurately without numerical instability or NaN propagation.
- [ ] Output feature tensor shapes match the expected input contract of the EchoGesture model architecture.
