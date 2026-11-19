---
name: sequence-model-training
description: >-
  Use this skill when implementing, configuring, and training GRU or Temporal CNN models for temporal sign recognition.
---

# Sequence Model Training for EchoGesture

## Purpose
This skill guides the implementation, training, hyperparameter tuning, and evaluation of sequential deep learning models (such as Gated Recurrent Units (GRUs) and Temporal Convolutional Networks (TCNs)) designed for recognizing temporal sign language gestures in the EchoGesture project.

## When to Activate
Activate this skill whenever you are:
- Designing, modifying, or instantiating GRU or Temporal CNN architecture modules for gesture sequence classification.
- Configuring dataset loaders, padding/truncation strategies, or temporal batching for variable-length sign sequences.
- Writing training loops, loss functions (e.g., Cross-Entropy, CTC loss), optimizer setups, or learning rate schedulers.
- Evaluating model checkpoints, running inference benchmarks, or debugging temporal alignment and vanishing/exploding gradients.

## Technical Workflow
1. **Data Preparation & Sequence Alignment**: Load extracted feature sequences (e.g., MediaPipe landmark embeddings or CNN spatial features), apply zero-padding or length-bucketing, and verify tensor shape conventions (`[batch_size, sequence_length, feature_dim]`).
2. **Model Architecture Instantiation**: Build or configure the sequence network (GRU layers with bidirectional processing and dropout, or TCN blocks with causal dilated convolutions and residual connections) adhering to EchoGesture modular design standards.
3. **Training Configuration & Optimization**: Set up loss criteria, AdamW optimizer with weight decay, gradient clipping to prevent explosion in recurrent/temporal networks, and cosine annealing learning rate schedules.
4. **Execution & Checkpointing**: Run training epochs with mini-batching, monitor validation loss and Top-1/Top-5 sign classification accuracy, and automatically save the best-performing model weights based on validation metrics.
5. **Evaluation & Artifact Logging**: Evaluate final checkpoints on the test split, generate confusion matrices across sign vocabulary classes, and verify inference latency for real-time responsiveness.

## Rules
- **Explicit Shapes & Types**: Always annotate tensor shapes and use explicit type hints across data loaders, model forward passes, and training step functions.
- **Gradient Stability**: Enforce gradient clipping (e.g., max norm of 1.0) during sequence model training to guard against exploding gradients in RNN/GRU backpropagation.
- **No Placeholder Code**: Provide complete, fully implemented training scripts, model modules, and configuration files without shortcuts or truncation comments.
- **Evidence of Execution**: Verify training runs and validation metrics by inspecting terminal output logs before declaring model implementation complete.

## Validation Checklist
- [ ] Tensor shape assertions and mask handling are correctly implemented for variable-length gesture sequences.
- [ ] GRU or TCN model modules are modular, parameterized, and free of hardcoded magic numbers.
- [ ] Training loops include proper gradient clipping, learning rate scheduling, and validation loss monitoring.
- [ ] Checkpoints correctly save model weights, optimizer states, and configuration metadata.
- [ ] Training and evaluation logs confirm successful convergence without NaN losses or memory leaks.
- [ ] Inference latency meets real-time sign recognition constraints on target hardware.
