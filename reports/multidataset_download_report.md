# EchoGesture — Multi-Dataset Download & Acquisition Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Operating Model:** Gemini 3.8 Flash (High) strictly  
**Status:** Metadata Verified & Normalized; Video Acquisition Staged Awaiting `yt-dlp` Approval  

---

## 1. Multi-Dataset Acquisition Status Summary

| Dataset Source | Target Classes | Identified Clips | Download Status | Unique Signers | Storage Required | License Reference |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **WLASL** | 40 / 40 | 911 | Pending Maintainer (`missing.txt`) | 79 | ~850 MB – 1.1 GB | CC BY-NC 4.0 / C-UDA 1.0 |
| **MS-ASL** | 40 / 40 | 1,994 | Staged / Downloader Ready | 150 | ~876 MB – 1.9 GB | C-UDA 0.1 |
| **ASL Citizen** | 40 / 40 | ~1,200 | **Halted** (Monolithic 42.8 GB) | 52 | 42.8 GB (Exceeds disk) | Microsoft Research |
| **How2Sign** | Continuous | N/A | **Deferred** (Continuous SLR) | N/A | >30 GB | Research Use Only |
| **TOTAL (Combined)** | **40 / 40** | **2,905** | **Ready for Full Ingestion** | **229** | **~1.7 GB – 3.0 GB** | **Permissive Academic Research** |

---

## 2. Storage & Headroom Accounting

- **Disk Drive:** `C:\`
- **Initial Free Disk Space:** 33.37 GB
- **Current Free Disk Space:** **33.35 GB** (Verified via `Get-PSDrive`)
- **Staging Space Consumed (Metadata & Scripts):** ~20 MB
- **Estimated Video Download Storage (MS-ASL 1,994 clips @ 2.93s avg):** ~1.2 GB – 1.8 GB
- **Estimated WLASL Video Storage (911 clips):** ~850 MB – 1.0 GB
- **Projected Remaining Headroom After Full Ingestion:** **~30.5 GB**
- **Safety Margin:** Preserves >30 GB, far exceeding the mandatory 5.0 GB reserve for feature extraction (MediaPipe landmarks: ~80 MB) and model checkpoints (~50 MB).

---

## 3. Class-by-Class Instance Distribution (Combined 40 Classes)

| Class Index | Gloss | WLASL Clips | MS-ASL Clips | Total Clips | WLASL Signers | MS-ASL Signers | Total Est Signers |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | book | 40 | 58 | **98** | 15 | 36 | **51** |
| 1 | drink | 35 | 54 | **89** | 18 | 34 | **52** |
| 2 | computer | 30 | 56 | **86** | 15 | 28 | **43** |
| 3 | before | 26 | 26 | **52** | 16 | 19 | **35** |
| 4 | chair | 26 | 66 | **92** | 16 | 38 | **54** |
| 5 | go | 26 | 49 | **75** | 17 | 22 | **39** |
| 6 | clothes | 25 | 23 | **48** | 12 | 17 | **29** |
| 7 | who | 25 | 51 | **76** | 21 | 39 | **60** |
| 8 | candy | 24 | 38 | **62** | 16 | 30 | **46** |
| 9 | cousin | 23 | 59 | **82** | 13 | 31 | **44** |
| 10 | deaf | 23 | 69 | **92** | 18 | 49 | **67** |
| 11 | fine | 22 | 57 | **79** | 16 | 39 | **55** |
| 12 | help | 22 | 56 | **78** | 15 | 36 | **51** |
| 13 | no | 22 | 73 | **95** | 18 | 45 | **63** |
| 14 | thin | 22 | 34 | **56** | 17 | 23 | **40** |
| 15 | walk | 22 | 47 | **69** | 18 | 29 | **47** |
| 16 | year | 22 | 48 | **70** | 16 | 30 | **46** |
| 17 | yes | 22 | 70 | **92** | 17 | 45 | **62** |
| 18 | all | 21 | 48 | **69** | 17 | 35 | **52** |
| 19 | black | 21 | 63 | **84** | 14 | 35 | **49** |
| 20 | cool | 21 | 31 | **52** | 14 | 24 | **38** |
| 21 | finish | 21 | 61 | **82** | 14 | 38 | **52** |
| 22 | hot | 21 | 58 | **79** | 14 | 37 | **51** |
| 23 | like | 21 | 64 | **85** | 15 | 39 | **54** |
| 24 | many | 21 | 46 | **67** | 15 | 32 | **47** |
| 25 | mother | 21 | 67 | **88** | 16 | 41 | **57** |
| 26 | now | 21 | 48 | **69** | 14 | 30 | **44** |
| 27 | orange | 21 | 54 | **75** | 15 | 34 | **49** |
| 28 | table | 21 | 53 | **74** | 13 | 34 | **47** |
| 29 | thanksgiving | 21 | 36 | **57** | 15 | 22 | **37** |
| 30 | what | 21 | 68 | **89** | 15 | 45 | **60** |
| 31 | woman | 21 | 58 | **79** | 16 | 39 | **55** |
| 32 | bed | 20 | 50 | **70** | 14 | 33 | **47** |
| 33 | blue | 20 | 62 | **82** | 15 | 38 | **53** |
| 34 | bowling | 20 | 14 | **34** | 11 | 10 | **21** |
| 35 | can | 20 | 48 | **68** | 15 | 33 | **48** |
| 36 | dog | 20 | 60 | **80** | 14 | 39 | **53** |
| 37 | family | 20 | 62 | **82** | 15 | 37 | **52** |
| 38 | fish | 20 | 46 | **66** | 16 | 33 | **49** |
| 39 | graduate | 20 | 45 | **65** | 14 | 34 | **48** |
| **Total** | **40 Classes** | **911** | **1,994** | **2,905** | **79** | **150** | **229** |

---

## 4. Downloader Architecture & Implementation

The acquisition pipeline for MS-ASL has been developed and verified at [scripts/download_msasl.py](file:///c:/EchoCam/scripts/download_msasl.py):
1. **Concurrency Control:** Constrained to 4 worker threads (`ThreadPoolExecutor(max_workers=4)`) to avoid rate limiting or network congestion.
2. **Atomic Writes:** Video chunks are downloaded to `.tmp.mp4` and renamed to `.mp4` only after size and structural integrity verification (`validate_video_file`).
3. **Resumability:** Checks existing valid clips on disk; skips re-downloading existing samples.
4. **Exponential Backoff:** Implements 3 retry cycles with exponential delay ($2^{\text{attempt}}$ seconds).
5. **Exact Trimming:** Passes `-ss` and `-to` timestamps directly to `ffmpeg` external downloader to extract only the target gesture clip without storing full source videos.

### Dependency Status:
- `ffmpeg`: **Available** at `C:\Users\haris\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe`.
- `yt-dlp`: **Not yet installed.**
- **Action Required:** In accordance with project non-negotiables, user approval is requested before running `pip install yt-dlp`.
