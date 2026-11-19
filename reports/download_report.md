# EchoGesture — Phase 2.6: Dataset Acquisition & Access Audit Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Status:** Pre-Download Verification Completed (Halted on Authorized Access Check)  
**Lead Model:** Gemini 3.8 Flash (High)  

---

## 1. Executive Summary & Stop-Condition Trigger

In accordance with the project's strict non-negotiable instructions:
> *"If the official access method requires a request or approval that has not yet been obtained, STOP and tell me what is required. Do not bypass it or substitute an unauthorized mirror."*
> *"STOP CONDITIONS: If access is unauthorized, credentials are required, the download source is unavailable, or the script encounters unexpected behavior, stop and report the issue."*

Our pre-download investigation has completed. **Bulk video downloads have not been executed.** 

This stop condition was triggered by two critical technical and authorized-access findings:
1. **Official Pre-Processed Archive Access:** The official, author-sanctioned method to obtain complete, pre-trimmed WLASL videos requires submitting the [Official WLASL Video Request Form](https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link) and agreeing to the `C-UDA-1.0` (Computational Use of Data Agreement) terms. We require confirmation whether the user has already obtained this authorized archive link.
2. **Web Scraping Attrition & Dependency Restriction:** The alternative method (scraping the 911 raw URLs listed in `WLASL_v0.3.json`) is blocked from downloading **294 YouTube clips (32.3%)** because `yt-dlp` is not installed on the system, and project rules strictly prohibit installing new dependencies without prior approval. Furthermore, empirical probing revealed that legacy educational sites (e.g., SigningSavvy, Handspeak) suffer from link decay (HTTP 404 / domain migrations).

---

## 2. Pre-Download Verification Checklist (Tasks 1–7)

| Verification Item | Audit Findings | Compliance Status |
| :--- | :--- | :---: |
| **1. Data-Use Agreement** | `C-UDA-1.0.pdf` (Computational Use of Data Agreement). Author grants rights to use data for AI model training and non-commercial research. | ✅ VERIFIED |
| **2. Authorized Access Method** | Two official channels: (A) Dynamic scraping via original URLs in `WLASL_v0.3.json`; (B) Clean pre-trimmed archive on request via Google Form. | ⚠️ AWAITING USER INPUT |
| **3. Legitimate Source URLs** | All 911 URLs in `data/splits/manifest.csv` come directly from the official `WLASL_v0.3.json` metadata authored by Li et al. (WACV 2020). | ✅ VERIFIED |
| **4. Error Handling & Validation** | `scripts/download_dataset.py` validates MP4 container headers (`ftyp`/`moov`), checks byte size, and writes atomic `.part` files. | ✅ VERIFIED |
| **5. Non-Destructive Downloads** | Existing valid video files in `data/raw/` are verified and automatically skipped; cannot be overwritten. | ✅ VERIFIED |
| **6. Resume Capabilities** | Interrupted downloads clean `.part` fragments; script skips already-completed valid video IDs on re-run. | ✅ VERIFIED |
| **7. Storage Headroom** | 911 video clips require **~850 MB – 1.0 GB** total. Workstation has **~39.8 GB free on `C:`** (>92% headroom preserved). | ✅ VERIFIED |

---

## 3. Manifest URL Domain & Accessibility Breakdown

Analysis of all 911 video entries in `data/splits/manifest.csv`:

| Hosting Domain / Source | Clip Count | Percentage | Acquisition Method & Current Status |
| :--- | :---: | :---: | :--- |
| **YouTube (`youtube.com` / `youtu.be`)** | 294 clips | 32.3% | Requires `yt-dlp` tool. `yt-dlp` is currently **NOT installed**. Blocked by dependency restriction. |
| **StartASL (AWS S3 Bucket)** | 45 clips | 4.9% | Direct HTTP streaming verified (**HTTP 200 OK**). |
| **SignSchool (Azure Blob Storage)** | 83 clips | 9.1% | Direct HTTP streaming verified (**HTTP 200 OK**). |
| **ASLDeafined (`asldeafined.com`)** | 57 clips | 6.3% | Direct HTTP streaming verified (**HTTP 200 OK**). |
| **SigningSavvy (`signingsavvy.com`)** | 104 clips | 11.4% | Direct HTTP returns **HTTP 404** (site changed internal storage structure since 2020). |
| **Handspeak (`handspeak.com`)** | 81 clips | 8.9% | Direct HTTP returns **DNS resolution failure / domain redirect**. |
| **ASLPro (`aslpro.com`)** | 91 clips | 10.0% | Legacy Flash/SWF/MP4 references; partial availability. |
| **Other Repositories (Yale, SpreadTheSign, etc.)** | 156 clips | 17.1% | Mixed HTTP availability across institutional repositories. |
| **TOTAL** | **911 clips** | **100.0%** | **67.7% Direct HTTP / 32.3% YouTube** |

---

## 4. Current File Status in `data/raw/`

```
Directory: C:\EchoCam\data\raw\
Existing Video Files: 0
Total Required:       911
Remaining to Acquire: 911
```

---

## 5. Two Paths to Proceed (User Decision Required)

### Option 1: User Imports Official Pre-Processed WLASL Archive (Recommended)
If you have already submitted the official Google Form and received the author's Google Drive archive, or if you have a local directory containing the pre-processed WLASL videos:
- Command:
  ```powershell
  python scripts/download_dataset.py --import-dir "C:\path\to\wlasl_videos" --output-dir data/raw
  ```
- **Advantages:** 100% of the 911 clips are acquired immediately with zero link rot, zero YouTube rate-limiting, and zero missing classes.

### Option 2: Approve Installing `yt-dlp` and Running Scripted Acquisition
If you wish to scrape the original public URLs directly:
1. Approve installing `yt-dlp` in the virtual environment.
2. Run `scripts/download_dataset.py` with 4 parallel worker threads.
3. Any clips that return 404 or are private/deleted will be logged to `reports/download_failures.log`. We will then audit the resulting class counts to confirm if each class retains sufficient sample density ($\ge 15$ clips).

---

*Halted in compliance with project stop conditions. Awaiting user guidance on Option 1 vs. Option 2.*
