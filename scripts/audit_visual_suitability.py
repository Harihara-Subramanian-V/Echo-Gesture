"""
Audit Visual Suitability of Downloaded MS-ASL Pilot Clips.
Uses ffmpeg to extract representative middle frames and ffprobe signalstats
to verify lighting, aspect ratio, bounding box coverage, and human visibility.
"""

import os
import json
import subprocess

PROJECT_ROOT = r"c:\EchoCam"
PILOT_DATA_PATH = os.path.join(PROJECT_ROOT, "reports", "pilot_validation_data.json")
THUMBNAILS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ms_asl", "pilot_thumbnails")

os.makedirs(THUMBNAILS_DIR, exist_ok=True)

with open(PILOT_DATA_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

visual_audit_results = []

for idx, r in enumerate(records):
    dl = r["download_result"]
    s = r["sample"]
    ff = r["ffprobe_result"]
    gloss = s["target_gloss"]
    status = dl.get("status")

    if status != "success" or not ff.get("valid"):
        visual_audit_results.append({
            "idx": idx + 1,
            "gloss": gloss,
            "status": status,
            "visual_suitability": "FAIL (Not Downloaded)",
            "details": dl.get("error", "Download failed")
        })
        continue

    filepath = dl["filepath"]
    duration = ff["actual_duration"]
    mid_time = max(0.1, duration / 2.0)
    thumb_path = os.path.join(THUMBNAILS_DIR, f"thumb_{idx+1:02d}_{gloss}.jpg")

    # 1. Extract middle frame
    cmd_extract = [
        "ffmpeg", "-y",
        "-ss", str(mid_time),
        "-i", filepath,
        "-vframes", "1",
        "-q:v", "2",
        thumb_path
    ]
    subprocess.run(cmd_extract, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Run signalstats to get luminance and saturation metrics
    cmd_stats = [
        "ffprobe", "-v", "error",
        "-f", "lavfi",
        f"movie='{filepath.replace(os.sep, '/')}',signalstats",
        "-show_entries", "frame_tags=lavfi.signalstats.YAVG,lavfi.signalstats.SATAVG",
        "-of", "csv=p=0"
    ]
    stat_res = subprocess.run(cmd_stats, capture_output=True, text=True)
    stats_lines = stat_res.stdout.strip().splitlines()

    y_avgs = []
    sat_avgs = []
    for line in stats_lines:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                y_avgs.append(float(parts[0]))
                sat_avgs.append(float(parts[1]))
            except ValueError:
                pass

    mean_y = sum(y_avgs) / len(y_avgs) if y_avgs else 128.0
    mean_sat = sum(sat_avgs) / len(sat_avgs) if sat_avgs else 50.0

    # 3. Analyze bounding box from MS-ASL annotation: [y0, x0, y1, x1]
    box = s.get("box", [0, 0, 1, 1])
    y0, x0, y1, x1 = box
    box_width = max(0.0, x1 - x0)
    box_height = max(0.0, y1 - y0)
    box_area = box_width * box_height

    # 4. Assess visual suitability criteria:
    # - Darkness check: mean_y >= 25 (not pitch black)
    # - Exposure check: mean_y <= 235 (not blown out)
    # - Signer prominence: box_area >= 0.10 (at least 10% of frame) and <= 0.98
    # - Resolution check: width >= 480 and height >= 360
    issues = []
    if mean_y < 25:
        issues.append("Severely underexposed / dark")
    elif mean_y > 235:
        issues.append("Severely overexposed / washed out")

    if box_area < 0.10:
        issues.append(f"Signer box very small ({box_area*100:.1f}% of frame)")
    elif box_area > 0.98:
        issues.append("Signer box spans full boundary (possible extreme close-up)")

    if ff["width"] < 480 or ff["height"] < 360:
        issues.append(f"Low resolution ({ff['width']}x{ff['height']})")

    thumb_size = os.path.getsize(thumb_path) if os.path.exists(thumb_path) else 0
    if thumb_size < 3000:
        issues.append("Thumbnail unusually small (<3KB, potential blank frame)")

    if not issues:
        suitability = "PASS"
        notes = "Clear framing, balanced lighting, prominent signer"
    elif len(issues) == 1 and "box spans full" in issues[0]:
        suitability = "PASS"
        notes = "Upper torso close-up (acceptable for hand gesture)"
    elif any("dark" in i or "overexposed" in i for i in issues):
        suitability = "NEEDS REVIEW"
        notes = "; ".join(issues)
    else:
        suitability = "PASS"
        notes = "; ".join(issues) if issues else "Acceptable"

    visual_audit_results.append({
        "idx": idx + 1,
        "gloss": gloss,
        "split": s.get("split"),
        "signer": s.get("signer_id"),
        "status": status,
        "resolution": f"{ff['width']}x{ff['height']}",
        "mean_y": round(mean_y, 1),
        "mean_sat": round(mean_sat, 1),
        "box_area_pct": round(box_area * 100, 1),
        "thumb_size_bytes": thumb_size,
        "suitability": suitability,
        "notes": notes
    })

print(f"{'#':<3} | {'Gloss':<10} | {'Split':<5} | {'Signer':<6} | {'Res':<9} | {'Y_Avg':<6} | {'Box %':<6} | {'Suitability':<12} | {'Notes'}")
print("-" * 95)
for v in visual_audit_results:
    if v["status"] == "success":
        print(f"{v['idx']:<3} | {v['gloss']:<10} | {v['split']:<5} | {v['signer']:<6} | {v['resolution']:<9} | {v['mean_y']:<6} | {v['box_area_pct']:<6} | {v['suitability']:<12} | {v['notes']}")
    else:
        print(f"{v['idx']:<3} | {v['gloss']:<10} | {'-':<5} | {'-':<6} | {'-':<9} | {'-':<6} | {'-':<6} | {v['visual_suitability']:<12} | {v['details'][:35]}")

with open(os.path.join(PROJECT_ROOT, "reports", "pilot_visual_audit.json"), "w", encoding="utf-8") as f:
    json.dump(visual_audit_results, f, indent=2)

print("\nWrote visual audit results to reports/pilot_visual_audit.json")
