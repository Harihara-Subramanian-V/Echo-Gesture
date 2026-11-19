"""
Append pilot failures to reports/multidataset_failures.log
"""

import json

with open(r"c:\EchoCam\reports\pilot_validation_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open(r"c:\EchoCam\reports\multidataset_failures.log", "a", encoding="utf-8") as flog:
    flog.write("\n# ==============================================================================\n")
    flog.write("# MS-ASL Pilot Acquisition Failures Log (20-Clip Pilot Run)\n")
    flog.write("# ==============================================================================\n")
    for d in data:
        dl = d["download_result"]
        if dl.get("status") == "failed":
            s = d["sample"]
            flog.write(f"[2026-11-19 17:11:00] [PILOT_FAILURE] Gloss: {s['target_gloss']:<10} | Split: {s.get('split'):<5} | Signer: {s['signer_id']} | URL: {s['url']} | Reason: {dl.get('error')}\n")

print("Appended pilot failures to reports/multidataset_failures.log")
