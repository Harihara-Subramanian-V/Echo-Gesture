"""
Parse pilot validation data and generate detailed summary tables.
"""

import os
import json

data_path = r"c:\EchoCam\reports\pilot_validation_data.json"
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total pilot samples: {len(data)}")
success_count = sum(1 for d in data if d["download_result"].get("status") == "success")
valid_count = sum(1 for d in data if d["ffprobe_result"].get("valid"))
print(f"Success: {success_count}, Valid: {valid_count}")

rows = []
for idx, d in enumerate(data):
    s = d["sample"]
    dl = d["download_result"]
    ff = d["ffprobe_result"]
    st = dl.get("status")
    gloss = s["target_gloss"]
    sp = s.get("split")
    sig = s.get("signer_id")
    url = s.get("url")
    dur_tgt = dl.get("target_duration", 0.0)
    dur_act = ff.get("actual_duration", None)
    delta = ff.get("duration_delta", None)
    res = f"{ff.get('width', 0)}x{ff.get('height', 0)}" if ff.get("valid") else "N/A"
    fps = ff.get("fps", "N/A")
    codec = ff.get("codec", "N/A")
    sz = f"{ff.get('size_bytes', 0) / 1024:.1f} KB" if ff.get("valid") else "N/A"
    err = dl.get("error", "")

    # Clean error description
    if "Private" in err or "private" in err:
        err_type = "Video Private"
    elif "unavailable" in err:
        err_type = "Video Unavailable"
    elif not err:
        err_type = "None (OK)"
    else:
        err_type = err[:40]

    rows.append({
        "idx": idx + 1,
        "gloss": gloss,
        "split": sp,
        "signer": sig,
        "status": st,
        "tgt_dur": round(dur_tgt, 2),
        "act_dur": round(dur_act, 2) if dur_act is not None else "N/A",
        "delta": round(delta, 3) if delta is not None else "N/A",
        "res": res,
        "fps": fps,
        "codec": codec,
        "size": sz,
        "err_type": err_type,
        "url": url
    })

print("\nPilot Clips Summary Table:")
print(f"{'#':<3} | {'Gloss':<10} | {'Split':<5} | {'Signer':<6} | {'Status':<7} | {'Tgt Dur':<7} | {'Act Dur':<7} | {'Delta':<7} | {'Res':<9} | {'Size':<9} | {'Notes'}")
print("-" * 95)
for r in rows:
    print(f"{r['idx']:<3} | {r['gloss']:<10} | {r['split']:<5} | {r['signer']:<6} | {r['status']:<7} | {str(r['tgt_dur']) + 's':<7} | {str(r['act_dur']) + ('s' if r['act_dur'] != 'N/A' else ''):<7} | {str(r['delta']):<7} | {r['res']:<9} | {r['size']:<9} | {r['err_type']}")
