# EchoGesture — Phase 2.5: Download & Acquisition Instructions
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Status:** Acquisition Protocols Prepared (Awaiting User Execution Approval)  
**Lead Model:** Gemini 3.8 Flash (High)  

---

## 1. Official Download Sources & Channel Verification

Following the project's strict non-negotiable rule against using unverified third-party mirrors or fabricating URLs, we inspected the official WLASL distribution repository (`dxli94/WLASL`):

| Resource | Official Location / Mechanism | Verification Status |
| :--- | :--- | :---: |
| **Official Repository** | `https://github.com/dxli94/WLASL` | Verified Active |
| **Official Metadata** | `start_kit/WLASL_v0.3.json` (11.4 MB) | Verified Active (HTTP 200) |
| **Official License** | `C-UDA-1.0.pdf` (Computational Use of Data Agreement) | Verified in Repo Root |
| **Video Acquisition Tool** | `start_kit/video_downloader.py` (uses `yt-dlp`) | Verified in `start_kit/` |
| **Pre-Processed Videos on Request** | Official Author Google Form: [WLASL Request Form](https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link) | Verified Active |
| **Direct Public Static `.zip` Archive** | **Does NOT exist on official portal / GitHub.** The authors intentionally distribute through dynamic download scripts and the Google Form to respect content source terms. | Verified Reality (No fake URL) |

---

## 2. Acquisition Strategies

To ensure 100% reproducibility while adapting to real-world network and platform constraints, EchoGesture provides two complementary acquisition routes:

### Route A: Targeted Scripted Download (Primary)
Uses our dedicated, quota-conscious downloader script:
```powershell
python scripts/download_dataset.py --manifest data/splits/manifest.csv --output-dir data/raw --max-workers 4
```

- **Selective Pull:** Downloads **only the 911 approved video IDs** from `data/splits/manifest.csv`.
- **Dual-Protocol Engine:**
  - Standard HTTP/HTTPS streaming with atomic file renaming (`.part` $\rightarrow$ `.mp4`) for direct video links (AWS S3, Azure Blob, Signing Savvy, Handspeak, StartASL).
  - `yt-dlp` backend for YouTube-hosted clips.
- **Fault-Tolerant Resumption:** Inspects existing files in `data/raw/` for valid MP4 headers; automatically skips already-downloaded clips.
- **Transparent Failure Logging:** Failed URLs (e.g. decayed YouTube links) are logged to `reports/download_failures.log` without crashing the batch.

### Route B: Direct Pre-Processed Archive Import (Alternative)
If you have access to the pre-trimmed video archive provided by the dataset authors via their official request form, you can import and filter it instantly:
```powershell
python scripts/download_dataset.py --import-dir "D:\path\to\extracted_wlasl_videos" --output-dir data/raw
```
- Reads the 911 approved IDs from `data/splits/manifest.csv`.
- Extracts and copies **only the 911 matching clips** into `data/raw/`.
- Verifies video container integrity and ignores unneeded clips from the other 1,960 classes, saving gigabytes of disk space.

---

## 3. Storage & Safety Guardrails

- **Storage Footprint:**
  - 911 raw MP4 clips: **~750 MB to 1.0 GB**.
  - Disk Headroom on `C:`: **~39.8 GB free** $\rightarrow$ **~38.8 GB free post-download** (>92% headroom preserved).
- **Execution Guard:**
  - `scripts/download_dataset.py` contains automated dry-run testing (`--dry-run`).
  - No files will be downloaded without your explicit instruction.

---

## 4. Next Step: Execution Approval

When you are ready to begin acquiring the 911 video clips, simply reply with your approval:
- To run automated acquisition: approve executing `scripts/download_dataset.py`.
- To import from a local folder: provide the folder path for `--import-dir`.
