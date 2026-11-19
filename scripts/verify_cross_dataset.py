"""
Verification of URL overlap, signer identity collision, and unified splits between WLASL and MS-ASL.
"""

import json
import csv
import re
from collections import defaultdict, Counter

wlasl_manifest = r"c:\EchoCam\data\splits\manifest.csv"
msasl_matched = r"c:\EchoCam\data\metadata\ms_asl\msasl_matched_samples.json"

# 1. Load WLASL samples
wlasl_data = []
with open(wlasl_manifest, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        wlasl_data.append(row)

# 2. Load MS-ASL samples
with open(msasl_matched, "r", encoding="utf-8") as f:
    msasl_data = json.load(f)

print(f"Total WLASL clips: {len(wlasl_data)}")
print(f"Total MS-ASL clips: {len(msasl_data)}")

# Helper to extract YouTube video ID
def extract_yt_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/embed\/|\/watch\?v=|youtu\.be\/|\/v\/)([a-zA-Z0-9_-]{11})", url)
    return m.group(1) if m else None

wlasl_yt_map = {}
wlasl_urls = {}
for r in wlasl_data:
    url = r.get("url", "")
    yt_id = extract_yt_id(url)
    if yt_id:
        wlasl_yt_map[yt_id] = r
    wlasl_urls[url] = r

msasl_yt_map = {}
msasl_urls = {}
for s in msasl_data:
    url = s.get("url", "")
    yt_id = extract_yt_id(url)
    if yt_id:
        msasl_yt_map[yt_id] = s
    msasl_urls[url] = s

print(f"WLASL YouTube videos: {len(wlasl_yt_map)}")
print(f"MS-ASL YouTube videos: {len(msasl_yt_map)}")

# Check YouTube ID intersection
common_yt_ids = set(wlasl_yt_map.keys()).intersection(set(msasl_yt_map.keys()))
print(f"\nYouTube ID Overlap between WLASL (40 classes) and MS-ASL (40 classes): {len(common_yt_ids)}")

if common_yt_ids:
    print("Found overlapping YouTube source videos! Examining splits and signers:")
    for yt_id in list(common_yt_ids)[:10]:
        w_sample = wlasl_yt_map[yt_id]
        m_sample = msasl_yt_map[yt_id]
        print(f"  YT ID: {yt_id}")
        print(f"    WLASL: gloss={w_sample['gloss']}, signer={w_sample['signer_id']}, split={w_sample['split']}, url={w_sample['url']}")
        print(f"    MS-ASL: gloss={m_sample['target_gloss']}, signer={m_sample['signer_id']}, split={m_sample['split']}, url={m_sample['url']}, time=({m_sample['start_time']}, {m_sample['end_time']})")

# Check all common YT IDs across splits
cross_split_leakage = []
for yt_id in common_yt_ids:
    w_split = wlasl_yt_map[yt_id]["split"]
    m_split = msasl_yt_map[yt_id]["split"]
    if w_split != m_split:
        cross_split_leakage.append((yt_id, w_split, m_split))

print(f"\nCross-dataset split alignment for shared YouTube videos: {len(common_yt_ids) - len(cross_split_leakage)} matching, {len(cross_split_leakage)} different")
if cross_split_leakage:
    print(f"Shared videos assigned to different splits across datasets: {cross_split_leakage}")

# Signer analysis
wlasl_signers = set(r["signer_id"] for r in wlasl_data)
msasl_signers = set(s["signer_id"] for s in msasl_data)
print(f"\nWLASL integer signer IDs: {sorted(list(int(s) for s in wlasl_signers))[:10]}...")
print(f"MS-ASL integer signer IDs: {sorted(list(int(s) for s in msasl_signers))[:10]}...")
