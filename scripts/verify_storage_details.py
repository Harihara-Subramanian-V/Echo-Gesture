"""
Detailed storage verification and technical stream inspection of downloaded pilot clips.
Computes exact byte sizes, average, median, min, max, resolution, codec, duration, fps,
and statistical storage projections for 1,994 candidate clips.
"""

import os
import json
import subprocess
import statistics

PROJECT_ROOT = r"c:\EchoCam"
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl")

files = [f for f in os.listdir(RAW_DIR) if f.startswith("msasl_pilot_") and f.endswith(".mp4")]
files.sort()

records = []
sizes = []

for f in files:
    fpath = os.path.join(RAW_DIR, f)
    size_bytes = os.path.getsize(fpath)
    sizes.append(size_bytes)

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=width,height,codec_name,r_frame_rate,duration,bit_rate,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        fpath
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout)

    v_stream = next((s for s in info.get("streams", []) if s.get("codec_name") != "aac"), {})
    a_stream = next((s for s in info.get("streams", []) if s.get("codec_name") == "aac"), {})
    fmt = info.get("format", {})

    fps_str = v_stream.get("r_frame_rate", "0/1")
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den != 0 else 0
    else:
        fps = float(fps_str)

    actual_dur = float(fmt.get("duration", v_stream.get("duration", 0.0)))
    bitrate = int(fmt.get("bit_rate", v_stream.get("bit_rate", 0)))
    w = v_stream.get("width")
    h = v_stream.get("height")
    res_str = f"{w}x{h}"

    records.append({
        "filename": f,
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024, 2),
        "duration": round(actual_dur, 3),
        "width": w,
        "height": h,
        "resolution": res_str,
        "fps": round(fps, 2),
        "v_codec": v_stream.get("codec_name"),
        "pix_fmt": v_stream.get("pix_fmt"),
        "a_codec": a_stream.get("codec_name", "none"),
        "bitrate_kbps": round(bitrate / 1000, 1)
    })

total_bytes = sum(sizes)
avg_bytes = statistics.mean(sizes)
median_bytes = statistics.median(sizes)
min_bytes = min(sizes)
max_bytes = max(sizes)
stdev_bytes = statistics.stdev(sizes) if len(sizes) > 1 else 0

print("Individual File Measurements:")
print(f"{'Filename':<45} | {'Size (KB)':<10} | {'Dur (s)':<8} | {'Resolution':<10} | {'FPS':<6} | {'Codec':<6} | {'Bitrate'}")
print("-" * 105)
for r in records:
    print(f"{r['filename']:<45} | {r['size_kb']:<10} | {r['duration']:<8} | {r['resolution']:<10} | {r['fps']:<6} | {r['v_codec']:<6} | {r['bitrate_kbps']} kbps")

print("\nSummary Statistics (14 Clips):")
print(f"  Total Bytes:   {total_bytes:,} bytes ({total_bytes / (1024*1024):.3f} MB)")
print(f"  Average Size:  {avg_bytes:,.1f} bytes ({avg_bytes / 1024:.2f} KB)")
print(f"  Median Size:   {median_bytes:,.1f} bytes ({median_bytes / 1024:.2f} KB)")
print(f"  Min Size:      {min_bytes:,} bytes ({min_bytes / 1024:.2f} KB) [{min(records, key=lambda x: x['size_bytes'])['filename']}]")
print(f"  Max Size:      {max_bytes:,} bytes ({max_bytes / 1024:.2f} KB) [{max(records, key=lambda x: x['size_bytes'])['filename']}]")
print(f"  Std Deviation: {stdev_bytes:,.1f} bytes ({stdev_bytes / 1024:.2f} KB)")

# Projections for 1,994 clips
proj_100_avg_mb = (1994 * avg_bytes) / (1024 * 1024)
proj_100_med_mb = (1994 * median_bytes) / (1024 * 1024)
proj_100_max_mb = (1994 * max_bytes) / (1024 * 1024)

# 70% retention projection (1,396 clips)
proj_70_avg_mb = (1396 * avg_bytes) / (1024 * 1024)
proj_70_med_mb = (1396 * median_bytes) / (1024 * 1024)

print("\nRealistic Projections for 1,994 MS-ASL Candidates:")
print(f"  1. 100% Retention (All 1,994 succeed):")
print(f"     - Mean Estimate:   {proj_100_avg_mb:.2f} MB (~{proj_100_avg_mb/1024:.3f} GB)")
print(f"     - Median Estimate: {proj_100_med_mb:.2f} MB (~{proj_100_med_mb/1024:.3f} GB)")
print(f"     - Worst Case (All clips at Max Size): {proj_100_max_mb:.2f} MB (~{proj_100_max_mb/1024:.3f} GB)")
print(f"  2. 70% Retention (~1,396 accessible clips, based on empirical 70% pilot rate):")
print(f"     - Mean Estimate:   {proj_70_avg_mb:.2f} MB (~{proj_70_avg_mb/1024:.3f} GB)")
print(f"     - Median Estimate: {proj_70_med_mb:.2f} MB (~{proj_70_med_mb/1024:.3f} GB)")

out_data = {
    "records": records,
    "summary": {
        "count": len(records),
        "total_bytes": total_bytes,
        "avg_bytes": avg_bytes,
        "median_bytes": median_bytes,
        "min_bytes": min_bytes,
        "max_bytes": max_bytes,
        "stdev_bytes": stdev_bytes,
        "projections": {
            "proj_100_avg_mb": round(proj_100_avg_mb, 2),
            "proj_100_med_mb": round(proj_100_med_mb, 2),
            "proj_100_max_mb": round(proj_100_max_mb, 2),
            "proj_70_avg_mb": round(proj_70_avg_mb, 2),
            "proj_70_med_mb": round(proj_70_med_mb, 2)
        }
    }
}
with open(os.path.join(PROJECT_ROOT, "reports", "pilot_storage_audit.json"), "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2)

print("\nWrote storage audit to reports/pilot_storage_audit.json")
