import json

with open("reports/sustained_memory_audit_results.json") as f:
    data = json.load(f)

print(f"Device: {data['device']}")
print(f"PID: {data['pid']}")
print(f"Sample Count: {data['sample_count']}")
print(f"Duration: {data['total_duration_sec']}s")

print("\n--- SAMPLE TIMELINE SUMMARY ---")
print(f"{'Idx':<4} {'Time(s)':<8} {'Phase':<22} {'PSS Tot(MB)':<12} {'Native(MB)':<12} {'Dalvik(MB)':<12} {'PrivDirty(MB)':<14} {'Bitmaps':<16} {'Views':<6}")
print("-" * 110)
for s in data["samples"]:
    t_mb = s['pss_total'] / 1024
    n_mb = s['pss_native'] / 1024
    d_mb = s['pss_dalvik'] / 1024
    pd_mb = s['private_dirty_total'] / 1024
    bmp_str = f"{s['bitmap_count']} ({s['bitmap_kb']}KB)"
    print(f"{s['sample_idx']:<4} {s['elapsed_sec']:<8.1f} {s['phase']:<22} {t_mb:<12.1f} {n_mb:<12.1f} {d_mb:<12.1f} {pd_mb:<14.1f} {bmp_str:<16} {s['views']:<6}")

print("\n--- RECENT GC LOGS ---")
for gc in data.get("recent_gc_events", []):
    print(" ", gc)
