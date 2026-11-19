"""
EchoGesture MS-ASL Bulk Clip Downloader and Trimmer (Batched & Checkpointed)
Downloads the 1,994 clips corresponding to the 40 EchoGesture classes in controlled batches.
Features:
- Batched execution with detailed checkpoint reports after every batch
- Concurrency control (default: 4 workers)
- Atomic file writes (.tmp.mp4 -> .mp4)
- Post-download ffprobe stream validation (rejects invalid/empty files)
- Strict disk space enforcement (halts if free disk space < 5.0 GB)
- Resume support (skips verified existing clips)
- Exponential backoff retry logic (up to 3 retries)
- Detailed failure logging to reports/multidataset_failures.log
"""

import os
import sys
import json
import time
import shutil
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
from yt_dlp.utils import download_range_func

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "msasl_matched_samples.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")
FAILURES_LOG_PATH = os.path.join(PROJECT_ROOT, "reports", "multidataset_failures.log")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "reports", "msasl_bulk_checkpoint.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(FAILURES_LOG_PATH, mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("MSASLBulkDownloader")

def check_free_space_gb() -> float:
    """Returns free space on drive C in gigabytes."""
    total, used, free = shutil.disk_usage("C:\\")
    return free / (1024 ** 3)

def check_dependencies():
    """Verify yt-dlp and ffmpeg availability."""
    missing = []
    try:
        import yt_dlp
    except ImportError:
        missing.append("yt-dlp (Python package: pip install yt-dlp)")

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        missing.append("ffmpeg (system executable)")

    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        missing.append("ffprobe (system executable)")

    return missing

def validate_with_ffprobe(filepath: str) -> bool:
    """Validate that the video exists, >10KB, and has a decodable video stream."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) <= 10240:
        return False
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            filepath
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return bool(res.stdout.strip())
    except Exception:
        return False

def download_and_trim_clip(sample: dict, index: int, max_retries: int = 3) -> dict:
    """
    Downloads and trims a single clip from YouTube using yt-dlp download_ranges.
    """
    # Enforce minimum 5 GB free disk space
    free_gb = check_free_space_gb()
    if free_gb < 5.0:
        raise RuntimeError(f"EMERGENCY HALT: Free disk space dropped to {free_gb:.2f} GB (< 5.0 GB threshold).")

    url = sample["url"]
    if not url.startswith("http"):
        url = "https://" + url

    start_time = float(sample.get("start_time", 0.0))
    end_time = float(sample.get("end_time", 0.0))
    split = sample.get("split", "train")
    gloss = sample.get("target_gloss", "unknown")

    yt_id = url.split("v=")[-1].split("&")[0] if "v=" in url else f"clip_{index:05d}"
    filename = f"msasl_{split}_{index:04d}_{yt_id}.mp4"
    final_path = os.path.join(OUTPUT_DIR, filename)
    temp_path = os.path.join(OUTPUT_DIR, f"{filename}.tmp.mp4")

    # Skip already downloaded and ffprobe-validated clips
    if validate_with_ffprobe(final_path):
        return {
            "status": "skipped_exists",
            "video_id": filename,
            "url": url,
            "path": final_path,
            "size_bytes": os.path.getsize(final_path),
            "valid": True
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_duration = max(0.1, end_time - start_time)

    for attempt in range(1, max_retries + 1):
        try:
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "outtmpl": temp_path,
                "download_ranges": download_range_func(None, [(start_time, end_time)]),
                "force_keyframes_at_cuts": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "ios", "mweb"]
                    }
                },
                "quiet": True,
                "no_warnings": True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Atomic validation with ffprobe
            if os.path.exists(temp_path) and validate_with_ffprobe(temp_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
                return {
                    "status": "success",
                    "video_id": filename,
                    "url": url,
                    "path": final_path,
                    "size_bytes": os.path.getsize(final_path),
                    "valid": True,
                    "attempt": attempt
                }
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError("Downloaded file failed ffprobe validation or was empty (<10KB).")

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            if attempt == max_retries:
                err_msg = f"Failed {filename} after {max_retries} attempts: {str(e)} | URL: {url}"
                logger.error(err_msg)
                return {
                    "status": "failed",
                    "video_id": filename,
                    "url": url,
                    "path": None,
                    "size_bytes": 0,
                    "valid": False,
                    "error": str(e),
                    "attempt": attempt
                }

            time.sleep(2 ** attempt)

    return {"status": "failed", "video_id": filename, "url": url, "path": None, "valid": False, "error": "Max retries exceeded"}

def run_batched_acquisition(batch_size: int = 250, max_workers: int = 4, target_batch: int = None):
    missing_deps = check_dependencies()
    if missing_deps:
        logger.error(f"Missing required dependencies: {missing_deps}")
        print("\n[ERROR] Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        return

    if not os.path.exists(METADATA_PATH):
        logger.error(f"Metadata file not found: {METADATA_PATH}")
        return

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    total_candidates = len(all_samples)
    num_batches = (total_candidates + batch_size - 1) // batch_size

    logger.info(f"Loaded {total_candidates} MS-ASL candidates. Partitioned into {num_batches} batches of up to {batch_size} clips.")

    # Load existing checkpoint if available
    checkpoint = {"batches": [], "cumulative": {"attempted": 0, "success": 0, "failed": 0, "skipped_exists": 0, "valid": 0, "invalid": 0, "bytes": 0}}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            pass

    start_batch = 0 if target_batch is None else target_batch
    end_batch = num_batches if target_batch is None else target_batch + 1

    for b_idx in range(start_batch, end_batch):
        b_start = b_idx * batch_size
        b_end = min(total_candidates, (b_idx + 1) * batch_size)
        batch_slice = all_samples[b_start:b_end]
        remaining = total_candidates - b_end

        free_gb = check_free_space_gb()
        print("\n" + "=" * 80)
        print(f"STARTING BATCH {b_idx + 1} / {num_batches} (Clips {b_start} to {b_end - 1}, Count: {len(batch_slice)})")
        print(f"Workers: {max_workers} | Available Disk Space: {free_gb:.2f} GB")
        print("=" * 80)

        batch_results = {"attempted": len(batch_slice), "success": 0, "failed": 0, "skipped_exists": 0, "valid": 0, "invalid": 0, "bytes": 0}
        b_start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(download_and_trim_clip, sample, b_start + i): b_start + i
                for i, sample in enumerate(batch_slice)
            }

            for future in as_completed(future_to_idx):
                res = future.result()
                st = res.get("status")
                sz = res.get("size_bytes", 0)
                is_valid = res.get("valid", False)

                if st == "success":
                    batch_results["success"] += 1
                    batch_results["valid"] += 1
                    batch_results["bytes"] += sz
                elif st == "skipped_exists":
                    batch_results["skipped_exists"] += 1
                    batch_results["valid"] += 1
                    batch_results["bytes"] += sz
                elif st == "failed":
                    batch_results["failed"] += 1
                    batch_results["invalid"] += 1

        b_elapsed = time.time() - b_start_time
        free_gb_after = check_free_space_gb()

        # Update cumulative counters
        checkpoint["cumulative"]["attempted"] += batch_results["attempted"]
        checkpoint["cumulative"]["success"] += batch_results["success"]
        checkpoint["cumulative"]["skipped_exists"] += batch_results["skipped_exists"]
        checkpoint["cumulative"]["failed"] += batch_results["failed"]
        checkpoint["cumulative"]["valid"] += batch_results["valid"]
        checkpoint["cumulative"]["invalid"] += batch_results["invalid"]
        checkpoint["cumulative"]["bytes"] += batch_results["bytes"]

        batch_summary_entry = {
            "batch_index": b_idx + 1,
            "range": [b_start, b_end - 1],
            "elapsed_seconds": round(b_elapsed, 1),
            "attempted": batch_results["attempted"],
            "new_success": batch_results["success"],
            "skipped_exists": batch_results["skipped_exists"],
            "total_valid": batch_results["valid"],
            "failed": batch_results["failed"],
            "bytes_consumed": batch_results["bytes"],
            "mb_consumed": round(batch_results["bytes"] / (1024 * 1024), 3),
            "free_space_gb": round(free_gb_after, 2),
            "remaining_candidates": remaining
        }
        checkpoint["batches"].append(batch_summary_entry)

        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        # Output detailed batch report to stdout
        print("\n" + "-" * 80)
        print(f"BATCH {b_idx + 1} COMPLETE in {b_elapsed:.1f}s")
        print(f"  - Attempted downloads:  {batch_results['attempted']}")
        print(f"  - Successful downloads: {batch_results['success']} new (plus {batch_results['skipped_exists']} verified existing)")
        print(f"  - Failed downloads:     {batch_results['failed']} (permanent link rot logged)")
        print(f"  - Valid videos (ffprobe): {batch_results['valid']}")
        print(f"  - Invalid videos:       {batch_results['invalid']}")
        print(f"  - Disk space consumed:  {batch_results['bytes'] / (1024*1024):.2f} MB (Free: {free_gb_after:.2f} GB)")
        print(f"  - Remaining candidates: {remaining}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("ALL BATCHES FINISHED!")
    print(f"Cumulative Summary across all batches:")
    print(f"  - Total Candidates Attempted: {total_candidates}")
    print(f"  - Total Valid Clips on Disk:  {checkpoint['cumulative']['valid']}")
    print(f"  - Total Failed (Link Rot):    {checkpoint['cumulative']['failed']}")
    print(f"  - Total Storage Consumed:     {checkpoint['cumulative']['bytes'] / (1024*1024):.2f} MB")
    print(f"  - Final Free Disk Space:      {check_free_space_gb():.2f} GB")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MS-ASL clips in controlled batches for EchoGesture.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrency (max 4 workers).")
    parser.add_argument("--batch-size", type=int, default=250, help="Clips per batch.")
    parser.add_argument("--batch-num", type=int, default=None, help="Optional: run a single batch index (0-indexed).")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only.")
    args = parser.parse_args()

    if args.check_deps:
        missing = check_dependencies()
        if missing:
            print("MISSING_DEPENDENCIES:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)
        else:
            print("ALL_DEPENDENCIES_AVAILABLE")
            sys.exit(0)

    run_batched_acquisition(batch_size=args.batch_size, max_workers=args.workers, target_batch=args.batch_num)
