"""
EchoGesture MS-ASL Pilot Runner & Validator
Selects 20 diverse clips across classes and signers, downloads them using 2 workers,
and rigorously validates each with ffprobe against annotation timestamps.
"""

import os
import sys
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
from yt_dlp.utils import download_range_func

PROJECT_ROOT = r"c:\EchoCam"
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "msasl_matched_samples.json")
PILOT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "pilot_samples.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "msasl_pilot_report.md")

def select_pilot_samples(samples, count=20):
    """
    Selects 20 clips maximizing diversity across:
    - Target classes (aims for 20 unique classes)
    - Signers (aims for 20 unique signers)
    - Splits (train, val, test representation)
    """
    selected = []
    seen_classes = set()
    seen_signers = set()

    # Targets: 10 train, 5 val, 5 test
    split_targets = {"train": 10, "val": 5, "test": 5}
    split_counts = {"train": 0, "val": 0, "test": 0}

    # Pass 1: Try to satisfy all constraints simultaneously
    for s in samples:
        c = s["target_gloss"]
        sig = s["signer_id"]
        sp = s.get("split", "train")

        if sp in split_targets and split_counts[sp] < split_targets[sp]:
            if c not in seen_classes and sig not in seen_signers:
                selected.append(s)
                seen_classes.add(c)
                seen_signers.add(sig)
                split_counts[sp] += 1
                if len(selected) == count:
                    break

    # Pass 2: If we still need more to reach 20, relax constraints slightly
    if len(selected) < count:
        for s in samples:
            c = s["target_gloss"]
            sig = s["signer_id"]
            sp = s.get("split", "train")
            if s not in selected:
                if c not in seen_classes or sig not in seen_signers:
                    selected.append(s)
                    seen_classes.add(c)
                    seen_signers.add(sig)
                    split_counts[sp] = split_counts.get(sp, 0) + 1
                    if len(selected) == count:
                        break

    return selected[:count]

def download_clip(sample, idx, max_retries=3):
    url = sample["url"]
    if not url.startswith("http"):
        url = "https://" + url

    start_time = float(sample.get("start_time", 0.0))
    end_time = float(sample.get("end_time", 0.0))
    split = sample.get("split", "train")
    gloss = sample.get("target_gloss", "unknown")

    yt_id = url.split("v=")[-1].split("&")[0] if "v=" in url else f"clip_{idx:05d}"
    filename = f"msasl_pilot_{split}_{idx:02d}_{gloss}_{yt_id}.mp4"
    final_path = os.path.join(OUTPUT_DIR, filename)
    temp_path = os.path.join(OUTPUT_DIR, f"{filename}.tmp.mp4")

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

            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 10240:
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
                return {
                    "status": "success",
                    "filename": filename,
                    "filepath": final_path,
                    "size_bytes": os.path.getsize(final_path),
                    "sample": sample,
                    "target_duration": target_duration,
                    "attempt": attempt
                }
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError("Downloaded file was empty or smaller than 10KB.")

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt == max_retries:
                return {
                    "status": "failed",
                    "filename": filename,
                    "filepath": None,
                    "error": str(e),
                    "sample": sample,
                    "target_duration": target_duration,
                    "attempt": attempt
                }
            time.sleep(2 ** attempt)

    return {"status": "failed", "filename": filename, "error": "Max retries exceeded", "sample": sample}

def validate_clip_with_ffprobe(filepath, target_duration):
    if not filepath or not os.path.exists(filepath):
        return {"valid": False, "error": "File does not exist"}

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,nb_frames:format=duration,size,bit_rate",
        "-of", "json",
        filepath
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)

        fmt = info.get("format", {})
        streams = info.get("streams", [])
        if not streams:
            return {"valid": False, "error": "No video stream found"}

        video_stream = streams[0]
        actual_duration = float(fmt.get("duration", 0.0))
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "unknown")
        fps_str = video_stream.get("r_frame_rate", "0/1")
        
        num, den = map(int, fps_str.split("/")) if "/" in fps_str else (int(fps_str), 1)
        fps = num / den if den != 0 else 0

        # Duration delta
        duration_delta = abs(actual_duration - target_duration)
        # Valid if duration is reasonably close (within 0.5s or 25% of cut due to keyframes)
        valid_duration = duration_delta < max(0.5, 0.25 * target_duration)

        return {
            "valid": True,
            "actual_duration": actual_duration,
            "target_duration": target_duration,
            "duration_delta": duration_delta,
            "duration_acceptable": valid_duration,
            "width": width,
            "height": height,
            "codec": codec,
            "fps": round(fps, 2),
            "size_bytes": int(fmt.get("size", os.path.getsize(filepath))),
            "bitrate": int(fmt.get("bit_rate", 0))
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def main():
    print(f"Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    pilot_samples = select_pilot_samples(samples, count=20)
    print(f"Selected {len(pilot_samples)} pilot clips:")
    for idx, s in enumerate(pilot_samples):
        print(f"  {idx+1:02d}. Gloss: {s['target_gloss']:<12} | Signer: {s['signer_id']:<3} | Split: {s.get('split'):<5} | Timestamps: ({s['start_time']}, {s['end_time']})")

    with open(PILOT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(pilot_samples, f, indent=2)

    print("\nStarting download of 20 pilot clips using 2 worker threads...")
    start_wall_time = time.time()

    download_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_clip, s, idx): idx for idx, s in enumerate(pilot_samples)}
        for future in as_completed(futures):
            res = future.result()
            download_results.append(res)
            st = res.get("status")
            gloss = res["sample"]["target_gloss"]
            fn = res.get("filename", "")
            print(f"  [{st.upper()}] {gloss} ({fn})")

    elapsed_time = time.time() - start_wall_time
    print(f"\nAll downloads completed in {elapsed_time:.1f} seconds.")

    # Validation phase
    print("\nValidating downloaded videos with ffprobe...")
    validation_records = []
    total_bytes = 0
    success_count = 0
    valid_count = 0

    for res in download_results:
        s = res["sample"]
        st = res["status"]
        if st == "success":
            success_count += 1
            fpath = res["filepath"]
            v_info = validate_clip_with_ffprobe(fpath, res["target_duration"])
            if v_info.get("valid") and v_info.get("duration_acceptable"):
                valid_count += 1
            total_bytes += os.path.getsize(fpath)
        else:
            v_info = {"valid": False, "error": res.get("error")}

        validation_records.append({
            "sample": s,
            "download_result": res,
            "ffprobe_result": v_info
        })

    print(f"\nPilot Summary:")
    print(f"  Downloaded successfully: {success_count} / {len(pilot_samples)}")
    print(f"  Valid videos (ffprobe): {valid_count} / {len(pilot_samples)}")
    print(f"  Total storage consumed: {total_bytes / (1024*1024):.2f} MB")

    # Output detailed JSON results
    with open(os.path.join(PROJECT_ROOT, "reports", "pilot_validation_data.json"), "w", encoding="utf-8") as f:
        json.dump(validation_records, f, indent=2)

    print(f"Saved validation data to reports/pilot_validation_data.json")

if __name__ == "__main__":
    main()
