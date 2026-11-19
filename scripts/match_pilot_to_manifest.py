"""
Match pilot samples to canonical manifest filenames and copy existing valid files.
"""

import os
import json
import shutil

PROJECT_ROOT = r"c:\EchoCam"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "msasl_matched_samples.json")
PILOT_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "ms_asl", "pilot_samples.json")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    all_samples = json.load(f)

with open(PILOT_PATH, "r", encoding="utf-8") as f:
    pilot_samples = json.load(f)

for p_idx, p in enumerate(pilot_samples):
    match_idx = None
    for idx, s in enumerate(all_samples):
        if s["url"] == p["url"] and s["start_time"] == p["start_time"] and s["target_gloss"] == p["target_gloss"] and s.get("split") == p.get("split"):
            match_idx = idx
            break

    url = p["url"]
    yt_id = url.split("v=")[-1].split("&")[0] if "v=" in url else f"clip_{match_idx:05d}"
    split = p.get("split", "train")
    canonical_name = f"msasl_{split}_{match_idx:04d}_{yt_id}.mp4"
    canonical_path = os.path.join(RAW_DIR, canonical_name)

    # find corresponding pilot file
    gloss = p["target_gloss"]
    pilot_name = f"msasl_pilot_{split}_{p_idx:02d}_{gloss}_{yt_id}.mp4"
    pilot_path = os.path.join(RAW_DIR, pilot_name)

    if os.path.exists(pilot_path):
        if not os.path.exists(canonical_path):
            shutil.copy2(pilot_path, canonical_path)
            print(f"Copied pilot file to canonical: {pilot_name} -> {canonical_name}")
        else:
            print(f"Canonical already exists: {canonical_name}")
    else:
        print(f"Pilot failed or not present: {pilot_name}")
