---
name: hackathon-submission
description: >-
  Use this skill when preparing the final EchoGesture hackathon submission, including predictions.jsonl, results.csv, README, and reproducibility verification.
---

# Hackathon Submission Skill

## Purpose
Guide the preparation, validation, and packaging of the final EchoGesture sign language recognition hackathon submission, ensuring all required output files (`predictions.jsonl`, `results.csv`), documentation (`README.md`), and reproducibility checks meet the hackathon evaluation standards.

## When to Activate
Activate this skill when the user is ready to finalize, package, or verify the EchoGesture project outputs for submission, generate evaluation predictions, or ensure end-to-end reproducibility of the sign language recognition pipeline.

## Technical Workflow
1. **Model Evaluation & Inference Generation**: Run inference on the test dataset using the best-performing EchoGesture model checkpoint to generate `predictions.jsonl` with proper formatting and confidence scores.
2. **Metrics Compilation**: Compute evaluation metrics (e.g., accuracy, Top-K accuracy, latency, model size) and save the summary to `results.csv`.
3. **Documentation & README Polish**: Complete the project `README.md` with clear architecture overview, installation instructions, model weights link, and benchmark results.
4. **Reproducibility Verification**: Execute a clean end-to-end dry-run verification script or test suite to ensure the submission builds, runs, and reproduces reported scores without missing dependencies or hardcoded local paths.
5. **Archive Packaging**: Bundle all submission assets into the required directory structure and verify file existence and integrity.

## Rules
- Never hardcode absolute file paths or use placeholder mock outputs in `predictions.jsonl` or `results.csv`.
- Ensure all metric claims in documentation are backed by inspected test run logs or generated `results.csv` data.
- Validate that all dependencies are explicitly declared in project configuration files for seamless reproduction.

## Validation Checklist
- [ ] `predictions.jsonl` is generated, correctly formatted, and validated against schema.
- [ ] `results.csv` contains accurate performance benchmarks and evaluation metrics.
- [ ] `README.md` includes clear setup, training, and inference instructions.
- [ ] End-to-end reproducibility verified with a clean run from scratch.
- [ ] No temporary debug files, API keys, or sensitive credentials included in the submission bundle.
- [ ] All code adheres to project architectural and linting standards.
