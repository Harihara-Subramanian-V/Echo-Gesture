---
name: tflite-optimization
description: >-
  Use this skill when converting trained models to TFLite, performing INT8 quantization, benchmarking inference latency, and validating quantized model accuracy.
---

# TFLite Optimization for EchoGesture

## Purpose
This skill defines the standardized workflow for converting PyTorch/TensorFlow trained sign language recognition models from the EchoGesture project into highly optimized TensorFlow Lite (TFLite) formats with INT8 quantization, ensuring low-latency inference on mobile and edge devices without sacrificing gesture recognition accuracy.

## When to Activate
- When exporting trained sign language classification or gesture detection models to `.tflite`.
- When applying post-training quantization (PTQ) or integer quantization (INT8) using representative calibration datasets.
- When benchmarking model inference latency, memory footprint, and CPU/NPU execution profiles on target mobile hardware.
- When validating quantized model accuracy against validation/test sign gesture datasets.

## Technical Workflow
1. **Model Export & Preparation**: Export the trained EchoGesture PyTorch model to ONNX or SavedModel format, ensuring input tensor shapes match real-time video frame dimensions (e.g., `[1, 3, 224, 224]` or MediaPipe landmark input vectors).
2. **Representative Dataset Calibration**: Collect a representative sample set of 100-300 preprocessed sign language video frames or landmark feature tensors from the EchoGesture dataset to accurately calibrate activation and weight ranges during quantization.
3. **INT8 Quantization & Conversion**: Run the TFLite Converter with full integer quantization (`FLOAT16` or `INT8` fallback), specifying integer input/output types to optimize execution on mobile accelerators.
4. **Latency & Profiling Benchmark**: Benchmark the generated `.tflite` model using the TFLite Benchmark Tool or Python interpreter on target hardware to measure execution time per inference pass and memory peak.
5. **Accuracy Validation**: Evaluate top-1 and top-5 sign classification accuracy of the quantized `.tflite` model on the EchoGesture test split to ensure accuracy degradation stays within acceptable limits (< 1%).

## Rules
- Always use representative datasets derived from actual EchoGesture training data for INT8 calibration to avoid catastrophic accuracy drop on sign gestures.
- Verify that input and output tensor quantization parameters (scale and zero-point) are correctly configured for downstream mobile integration.
- Never deploy a quantized `.tflite` model without running comparative latency and accuracy benchmarks against the baseline unquantized model.
- Keep model size and peak memory usage within mobile constraints (< 25MB for real-time mobile inference).

## Validation Checklist
- [ ] PyTorch model successfully exported to intermediate format without graph breaking.
- [ ] Representative calibration dataset compiled from valid EchoGesture video/landmark samples.
- [ ] TFLite model generated with full INT8 quantization and verified tensor types.
- [ ] Inference latency benchmarked and meets real-time target (< 30ms per frame).
- [ ] Quantized model accuracy validated on EchoGesture test dataset with < 1% drop.
- [ ] Output `.tflite` artifact saved to the designated `models/` directory.
