---
name: video-preprocessing
description: >-
  Use this skill when implementing video loading, frame extraction, sampling, resizing, and validation for the EchoGesture preprocessing pipeline.
---

# Video Preprocessing Skill

## Purpose
This skill defines the technical standards, workflows, and validation checks for the video preprocessing pipeline in the EchoGesture sign language recognition project. It ensures robust video ingestion, consistent frame extraction, temporal sampling, and spatial normalization before feature extraction and inference.

## When to Activate
Activate this skill when:
- Implementing or modifying video loading utilities using OpenCV (`cv2`) or PyTorch video libraries.
- Developing frame sampling, temporal sliding windows, or uniform sub-sampling strategies for sign language videos.
- Setting up spatial resizing, cropping, and normalization pipelines for neural network inputs.
- Validating video container integrity, resolution, frame rate, and duration bounds.

## Technical Workflow
1. **Video Ingestion & Validation**: Open the target video file using OpenCV, verify container integrity, and extract metadata (total frames, frames per second, resolution, and duration) to ensure it meets minimum thresholds.
2. **Temporal Sampling & Frame Extraction**: Extract frames sequentially or apply uniform/sliding window temporal sampling to match the target frame count required by the downstream EchoGesture model architecture.
3. **Spatial Normalization & Resizing**: Resize extracted frames to the canonical input resolution (e.g., 224x224) using bilinear/bicubic interpolation, maintaining aspect ratio with padding if necessary.
4. **Tensor Conversion & Normalization**: Convert pixel arrays to float tensors, normalize pixel values to standard ranges (e.g., [0, 1] or mean/std normalization), and structure tensors in `[C, T, H, W]` or batch formats.

## Rules
- Never hardcode video resolution, target frame counts, or sampling strides; load them from configuration files or pipeline parameters.
- Always release OpenCV video capture resources explicitly in a `finally` block or context manager to prevent memory and file descriptor leaks.
- Ensure all video loaders handle corrupted or missing video files gracefully by raising standardized exceptions rather than crashing silently.
- Validate tensor shapes and data types before passing preprocessed batches to feature extractors or models.

## Validation Checklist
- [ ] Video file existence and readability are verified prior to processing.
- [ ] Metadata (FPS, frame count, codec) is successfully extracted and logged.
- [ ] Temporal sampling correctly outputs the exact target frame count.
- [ ] Spatial resizing preserves aspect ratio and avoids image distortion.
- [ ] Pixel normalization constants match the expected model pre-training statistics.
- [ ] Unit tests cover edge cases such as zero-length videos, corrupted codecs, and variable frame rates.
