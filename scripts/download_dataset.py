"""
EchoGesture Dataset Downloader & Validator
Acquires and verifies video clips specified in the curated 40-class manifest.
DO NOT EXECUTE WITHOUT EXPLICIT USER APPROVAL.
"""

import os
import csv
import sys
import time
import argparse
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logger(log_file):
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logger = logging.getLogger("EchoGestureDownloader")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    
    return logger

def is_valid_video(file_path, min_bytes=10240):
    """
    Verifies that a video file exists, has sufficient byte size,
    and has a valid container header.
    """
    if not os.path.exists(file_path):
        return False
    size = os.path.getsize(file_path)
    if size < min_bytes:
        return False
    # Check for MP4 'ftyp' or common container signatures in first 32 bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
            if b"ftyp" in header or b"moov" in header or b"RIFF" in header or b"\x1a\x45\xdf\xa3" in header:
                return True
    except Exception:
        return False
    return False

def download_direct_url(url, dest_path, timeout=30):
    """
    Streams download of a direct video URL to destination.
    Uses temporary .part file to avoid saving corrupted partial files.
    """
    part_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    })
    
    with urllib.request.urlopen(req, timeout=timeout) as response, open(part_path, "wb") as out_file:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            
    # Atomically rename on complete
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(part_path, dest_path)
    return True

def download_youtube_url(url, dest_path, timeout=60):
    """
    Invokes yt-dlp to download YouTube video if installed.
    """
    import subprocess
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--format", "mp4/best[ext=mp4]/best",
        "--output", dest_path,
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return res.returncode == 0 and os.path.exists(dest_path)

def process_item(item, output_dir, logger, dry_run=False):
    v_id = item["video_id"]
    gloss = item["gloss"]
    url = item["url"]
    dest_path = os.path.join(output_dir, f"{v_id}.mp4")

    # Check if already present and valid
    if is_valid_video(dest_path):
        return {"video_id": v_id, "status": "SKIPPED_EXISTING", "path": dest_path}

    if dry_run:
        return {"video_id": v_id, "status": "DRY_RUN", "url": url}

    if not url or url == "":
        logger.warning(f"Video {v_id} ({gloss}): Empty URL in manifest.")
        return {"video_id": v_id, "status": "FAILED", "reason": "EMPTY_URL"}

    try:
        # Determine download backend based on URL
        if "youtube.com" in url or "youtu.be" in url:
            success = download_youtube_url(url, dest_path)
        else:
            success = download_direct_url(url, dest_path)

        if success and is_valid_video(dest_path):
            logger.info(f"Downloaded {v_id}.mp4 for gloss '{gloss}'")
            return {"video_id": v_id, "status": "SUCCESS", "path": dest_path}
        else:
            logger.error(f"Download invalid/corrupted for {v_id} ({gloss}) from {url}")
            return {"video_id": v_id, "status": "FAILED", "reason": "CORRUPTED_OR_EMPTY"}

    except Exception as e:
        logger.error(f"Download failed for {v_id} ({gloss}) from {url}: {str(e)}")
        # Clean up any partial files
        part = dest_path + ".part"
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
        return {"video_id": v_id, "status": "FAILED", "reason": str(e)}

def import_preprocessed_videos(source_dir, output_dir, manifest_path, logger):
    """
    Imports and verifies approved clips from a pre-extracted local author archive.
    """
    import shutil
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        approved_ids = set(row["video_id"] for row in reader)

    logger.info(f"Importing approved clips from {source_dir} to {output_dir}...")
    copied = 0
    missing = []
    
    for v_id in approved_ids:
        src = os.path.join(source_dir, f"{v_id}.mp4")
        dst = os.path.join(output_dir, f"{v_id}.mp4")
        if os.path.exists(src) and is_valid_video(src):
            if not is_valid_video(dst):
                shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(v_id)

    logger.info(f"Import complete: {copied}/{len(approved_ids)} verified clips imported.")
    if missing:
        logger.warning(f"Missing {len(missing)} clips in source directory.")
    return copied, missing

def main():
    parser = argparse.ArgumentParser(description="EchoGesture Dataset Acquisition & Validator")
    parser.add_argument("--manifest", default="data/splits/manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--output-dir", default="data/raw", help="Target output directory for videos")
    parser.add_argument("--log-file", default="reports/download_failures.log", help="Failure log file")
    parser.add_argument("--max-workers", type=int, default=4, help="Thread pool concurrency")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without downloading")
    parser.add_argument("--import-dir", default=None, help="Import clips from pre-extracted archive directory")
    args = parser.parse_args()

    logger = setup_logger(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.import_dir:
        import_preprocessed_videos(args.import_dir, args.output_dir, args.manifest, logger)
        return

    if not os.path.exists(args.manifest):
        logger.error(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        items = list(reader)

    logger.info(f"Loaded {len(items)} items from manifest {args.manifest}")
    if args.dry_run:
        logger.info("Running in DRY-RUN mode. No files will be downloaded.")

    start_time = time.time()
    results = {"SUCCESS": 0, "SKIPPED_EXISTING": 0, "FAILED": 0, "DRY_RUN": 0}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_item, item, args.output_dir, logger, args.dry_run): item for item in items}
        for future in as_completed(futures):
            res = future.result()
            st = res.get("status", "FAILED")
            results[st] = results.get(st, 0) + 1

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Acquisition Summary (Time: {elapsed:.2f}s):")
    logger.info(f"  Total Requested:     {len(items)}")
    logger.info(f"  Already Present:     {results.get('SKIPPED_EXISTING', 0)}")
    logger.info(f"  Newly Succeeded:     {results.get('SUCCESS', 0)}")
    logger.info(f"  Failed:              {results.get('FAILED', 0)}")
    if args.dry_run:
        logger.info(f"  Dry-Run Verified:    {results.get('DRY_RUN', 0)}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
