---
name: robustness-testing
description: >-
  Use this skill when testing model robustness under controlled perturbations such as coordinate noise, temporal resampling, and landmark dropout.
---

# Robustness Testing

## Purpose
The purpose of this skill is to systematically evaluate the sign language recognition models within the EchoGesture project against various real-world perturbations and failure modes. By introducing controlled coordinate noise, temporal resampling variations, and landmark dropout, this testing suite ensures that the gesture recognition pipeline remains resilient to sensor jitter, variable signing speeds, and occlusions.

## When to Activate
Activate this skill when:
- Evaluating newly trained sign language recognition models prior to deployment or release.
- Investigating model sensitivity to tracking noise or missing skeletal landmarks from MediaPipe/OpenPose.
- Verifying the stability of temporal sequence models across different frame rates and signing velocities.

## Technical Workflow
1. **Perturbation Configuration**: Load the validation dataset and initialize perturbation parameters, specifying noise standard deviations for coordinate jitter, scaling factors for temporal resampling, and dropout rates for landmark occlusion.
2. **Synthetic Data Generation**: Apply deterministic transformations to input skeletal sequences, generating perturbed test splits that simulate noisy camera feeds, dropped frames, or unsteady hand tracking.
3. **Model Evaluation & Inference**: Run batched inference across the perturbed validation splits using the target EchoGesture recognition model, recording classification accuracy, confidence scores, and failure modes.
4. **Resilience & Degradation Analysis**: Compute performance degradation curves relative to the baseline clean dataset to identify vulnerability thresholds and failure boundaries for specific sign classes.
5. **Reporting & Logging**: Export structured metrics, confusion matrices, and robustness scores to experiment tracking dashboards to guide downstream data augmentation and architectural improvements.

## Rules
- Never modify or overwrite the original unperturbed evaluation dataset files; all perturbations must be applied dynamically or saved to isolated test splits.
- Ensure all perturbation transformations are seeded properly to guarantee reproducibility across evaluation runs.
- Do not bypass verification checks; performance drops exceeding established thresholds must be investigated and documented with evidence from CLI test outputs.

## Validation Checklist
- [ ] Baseline accuracy on clean validation data is successfully established and recorded.
- [ ] Coordinate noise perturbation tests execute successfully across multiple noise variance levels.
- [ ] Temporal resampling variations (speed-up and slow-down) are properly validated without tensor shape mismatches.
- [ ] Landmark dropout scenarios are tested to simulate partial hand/pose occlusion.
- [ ] Quantitative robustness metrics and degradation curves are generated and logged.
- [ ] Terminal execution logs and test summaries are verified before completing the evaluation task.
