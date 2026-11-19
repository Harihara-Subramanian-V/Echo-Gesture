---
name: mediapipe-landmarks
description: >-
  Use this skill when extracting hand and body landmarks from video frames using MediaPipe, including handling detection failures and missing landmarks.
---

# MediaPipe Landmarks Skill (EchoGesture)

## Purpose
This skill governs the robust extraction, normalization, and handling of hand and body landmarks using Google MediaPipe specifically tailored for the **EchoGesture** sign language recognition project. It ensures reliable spatial feature extraction, consistent tensor formatting for sequence models, and graceful fallback behaviors when hands or pose joints are occluded or missing.

## When to Activate
- When writing or modifying MediaPipe extraction pipelines (`mp.solutions.hands`, `mp.solutions.pose`).
- When dealing with missing or unconfident landmark detections in video processing frames.
- When formatting extracted coordinate tensors for ingestion into EchoGesture's classifier models.

## Technical Workflow
1. **Frame Ingestion & Color Conversion**: Read incoming video frames from the EchoCam pipeline using OpenCV (`cv2`), converting color spaces from BGR to RGB as required by MediaPipe solutions.
2. **Landmark Detection & Confidence Filtering**: Invoke MediaPipe Hands and Pose detectors with configured confidence thresholds (`min_detection_confidence`, `min_tracking_confidence`), filtering out low-confidence predictions to prevent noisy feature injection.
3. **Spatial Normalization & Origin Centering**: Normalize extracted landmark coordinates relative to a robust anchor point (e.g., wrist or torso center) and scale by bounding box dimensions to ensure invariance to user distance and camera position.
4. **Missing Landmark Imputation & Zero-Filling**: Handle detection failures (when hands or body parts leave the frame) by applying deterministic fallback strategies, such as zero-filling missing coordinate arrays or carrying forward historical imputation flags.
5. **Tensor Serialization**: Package the processed landmarks into structured NumPy arrays or PyTorch tensors matching EchoGesture's expected input feature dimensions, appending sequence metadata for temporal modeling.

## Rules
- **No Silent Failures**: Never silently ignore detection dropouts; explicitly log or flag frames with missing landmark detections so downstream temporal models can handle occlusion states.
- **Strict BGR/RGB Conversion**: Always convert OpenCV frames from BGR to RGB before passing them to MediaPipe estimators to prevent incorrect landmark color-space inferences.
- **Complete Blocks & No Placeholders**: Provide fully implemented Python extractor functions without placeholder comments (`# TODO`) or truncated code blocks.
- **Type Safety**: Enforce strict Python type annotations (`typing.Optional`, `np.ndarray`, etc.) across all landmark processing and feature scaling utility functions.

## Validation Checklist
- [ ] Frames are correctly converted from BGR to RGB prior to MediaPipe ingestion.
- [ ] Detection and tracking confidence thresholds are explicitly configured.
- [ ] Landmark coordinates are normalized relative to standard anchor joints (e.g., wrists/hips).
- [ ] Missing or occluded detections are gracefully handled with zero-filling or imputation flags.
- [ ] Output tensors match the expected input shape and dimensionality for EchoGesture models.
- [ ] Unit tests verify landmark extraction correctness and robustness against missing frames.
