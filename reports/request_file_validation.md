# EchoGesture — WLASL Video Request File Validation Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Status:** Validated & Ready for Submission  
**File Created:** `c:\EchoCam\missing.txt`  
**Lead Model:** Gemini 3.8 Flash (High)  

---

## 1. Executive Summary

In preparation for submitting the official **WLASL Video Request Form**, a clean, validated `missing.txt` request file was generated containing exclusively the **911 video IDs** corresponding to our curated 40-class EchoGesture sign language lexicon.

The generated file was strictly audited against `data/splits/manifest.csv` and cross-referenced with the official WLASL metadata (`WLASL_v0.3.json`).

---

## 2. File Specification & Verification Metrics

| Verification Check | Target / Requirement | Measured Result | Audit Status |
| :--- | :---: | :---: | :---: |
| **Target File Path** | Project root: `missing.txt` | `c:\EchoCam\missing.txt` | ✅ PASSED |
| **Total Lines / IDs** | Exactly 911 | **911** | ✅ PASSED |
| **Unique IDs** | Exactly 911 | **911** | ✅ PASSED |
| **Duplicate IDs** | Exactly 0 | **0** | ✅ PASSED |
| **Official WLASL Existence** | 100% of IDs in `WLASL_v0.3.json` | **911 / 911 (100.0%)** | ✅ PASSED |
| **IDs Outside Manifest** | Exactly 0 | **0** | ✅ PASSED |
| **Formatting Standard** | One ID per line, no header, no extension | Clean single-column string IDs | ✅ PASSED |
| **File Encoding & Linebreak** | Plain text (`UTF-8`), Unix `\n` | Validated | ✅ PASSED |
| **File Size on Disk** | ~5 – 6 KB | **5466 bytes** | ✅ PASSED |

### First 5 and Last 5 Entries
- **First 5 IDs:** `07068, 07069, 07070, 07071, 07072`
- **Last 5 IDs:** `25339, 65843, 67722, 68528, 70176`

---

## 3. Split Breakdown of Requested Video IDs

The 911 IDs in `missing.txt` map directly to the deterministic signer-independent splits established in Phase 2.5:

| Dataset Split | Clip Count | Percentage of Request | Gloss Coverage | Signer Independence |
| :--- | :---: | :---: | :---: | :---: |
| **Training Split** | 635 clips | 69.7% | 40 / 40 classes | 53 unique signers |
| **Validation Split** | 146 clips | 16.0% | 40 / 40 classes | 7 unique signers |
| **Testing Split** | 130 clips | 14.3% | 40 / 40 classes | 19 unique signers |
| **TOTAL** | **911 clips** | **100.0%** | **40 / 40 classes** | **79 signers (0 overlap)** |

---

## 4. Assessment on Maintainer Acceptance of Subset Requests

### Official Workflow Context
The official WLASL repository instructions state:
> *"Submit a video request by agreeing to terms of use at: https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link. You will get links to the missing videos within 7 days..."*

### Maintainer Response Expectations & Uncertainty Analysis:
1. **Google Form Processing Behavior:**
   - The form provides an upload dialog for `missing.txt`.
   - The maintainers do not typically filter and zip a custom bundle of 911 videos on-the-fly for each applicant.
   - Instead, the author (Dongxu Li) typically verifies the academic/research affiliation, confirms agreement to the `C-UDA 1.0` license, and shares access to a Google Drive folder hosting the pre-processed zip archives (e.g., `WLASL100.zip`, `WLASL300.zip`, or the complete `videos.zip`).
2. **Acceptance of Subset `missing.txt`:**
   - Submitting our 911-line `missing.txt` is completely valid and clearly indicates to the author that our research is constrained to a curated subset of high-frequency words.
   - If the author provides access to the full or standard subset Google Drive folder, our pre-built script:
     ```powershell
     python scripts/download_dataset.py --import-dir "<path_to_downloaded_archive>" --output-dir data/raw
     ```
     will automatically extract and import **only the 911 matching clips**, ensuring that no unnecessary videos are kept in local storage.

---

## 5. Next Steps for the User

1. Navigate to the [Official WLASL Video Request Form](https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link).
2. Enter your email and research/institutional affiliation.
3. Agree to the `C-UDA 1.0` Computational Use of Data Agreement.
4. Upload `c:\EchoCam\missing.txt`.
5. Once access is received from the author, use our import tool to populate `data/raw/`.
