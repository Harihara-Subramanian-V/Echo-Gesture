---
name: dataset-auditor
description: >-
  Use this skill when evaluating, selecting, or auditing sign language datasets (WLASL, How2Sign) for the EchoGesture project, including vocabulary selection and split strategy.
---

# Dataset Auditor Skill for EchoGesture

## Purpose
The `dataset-auditor` skill provides structured guidelines and workflows for evaluating, selecting, auditing, and preprocessing sign language datasets (such as WLASL and How2Sign) tailored specifically to the EchoGesture sign language recognition pipeline. It ensures consistent vocabulary subsetting, signer-independent data splits, and rigorous quality control over video frames and pose skeleton extractions.

## When to Activate
Activate this skill when:
- Selecting or filtering subsets of sign language vocabulary (e.g., top-100 or top-300 glosses from WLASL).
- Designing or validating signer-independent train/validation/test split strategies to prevent data leakage.
- Auditing raw video frames, bounding boxes, or MediaPipe pose/hand landmark keypoint extractions for missing data, occlusions, or frame-rate mismatches.
- Verifying dataset alignment with EchoGesture model input specifications.

## Technical Workflow
1. **Vocabulary Filtering & Subset Definition**: Load raw gloss annotations from WLASL/How2Sign, filter for minimum instance counts per gloss, and map gloss identifiers to EchoGesture's standardized taxonomy.
2. **Signer-Independent Splitting**: Partition the dataset ensuring that signers present in the validation and test sets do not overlap with the training set, eliminating signer bias and data leakage.
3. **Keypoint & Frame Auditing**: Run validation checks across extracted video frames and skeletal keypoint files (`.npz` / `.json`), flagging corrupted videos, empty keypoint arrays, or incorrect framerates.
4. **Dataset Statistics & Manifest Generation**: Compute final class distribution metrics, video duration statistics, and generate the definitive JSON manifest consumed by the EchoGesture dataloader.

## Rules
- Never allow overlapping signers between training, validation, and test splits.
- Do not use placeholder paths or unverified external dataset downloads; all source paths must be explicitly validated against local environment storage.
- Always log audit warnings and drop ratios (percentage of discarded samples due to missing keypoints) to ensure transparent preprocessing reproducibility.

## Validation Checklist
- [ ] Verified signer ID uniqueness across train, val, and test splits to prevent data leakage.
- [ ] Confirmed minimum instance counts per gloss meet the EchoGesture threshold.
- [ ] Audited skeletal landmark output files for NaN values, missing frames, or incorrect tensor shapes.
- [ ] Generated and validated the final dataset manifest JSON file structure.
- [ ] Inspected class distribution imbalance and documented mitigation strategies (e.g., class weighting or augmentation).
