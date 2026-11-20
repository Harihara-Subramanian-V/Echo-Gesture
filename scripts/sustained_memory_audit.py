import subprocess
import time
import re
import json
import os
import sys

ADB = r"C:\Users\haris\AppData\Local\Android\Sdk\platform-tools\adb.exe"
DEVICE = "10BF3B13J4004R5"
PKG = "com.echogesture.app"

def run_adb(cmd_args):
    full_cmd = [ADB, "-s", DEVICE] + cmd_args
    res = subprocess.run(full_cmd, capture_output=True, text=True, errors="replace")
    return res.stdout.strip()

def get_pid():
    out = run_adb(["shell", "pidof", PKG])
    pids = out.split()
    return pids[0] if pids else None

def parse_meminfo(pid):
    out = run_adb(["shell", "dumpsys", "meminfo", pid])
    metrics = {
        "pid": pid,
        "pss_total": 0,
        "pss_native": 0,
        "pss_dalvik": 0,
        "private_dirty_total": 0,
        "private_dirty_native": 0,
        "private_dirty_dalvik": 0,
        "heap_size_native": 0,
        "heap_alloc_native": 0,
        "heap_free_native": 0,
        "heap_size_dalvik": 0,
        "heap_alloc_dalvik": 0,
        "heap_free_dalvik": 0,
        "rss_total": 0,
        "bitmap_count": 0,
        "bitmap_kb": 0,
        "other_malloc_count": 0,
        "other_malloc_kb": 0,
        "views": 0,
        "activities": 0,
        "assets": 0,
        "raw_output": out
    }
    
    # Native Heap
    m_native = re.search(r"Native Heap\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if m_native:
        metrics["pss_native"] = int(m_native.group(1))
        metrics["private_dirty_native"] = int(m_native.group(2))
        metrics["heap_size_native"] = int(m_native.group(3))
        metrics["heap_alloc_native"] = int(m_native.group(4))
        metrics["heap_free_native"] = int(m_native.group(5))

    # Dalvik Heap
    m_dalvik = re.search(r"Dalvik Heap\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if m_dalvik:
        metrics["pss_dalvik"] = int(m_dalvik.group(1))
        metrics["private_dirty_dalvik"] = int(m_dalvik.group(2))
        metrics["heap_size_dalvik"] = int(m_dalvik.group(3))
        metrics["heap_alloc_dalvik"] = int(m_dalvik.group(4))
        metrics["heap_free_dalvik"] = int(m_dalvik.group(5))

    # TOTAL
    m_tot = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\d+)", out)
    if m_tot:
        metrics["pss_total"] = int(m_tot.group(1))
        metrics["private_dirty_total"] = int(m_tot.group(2))
        metrics["rss_total"] = int(m_tot.group(3))

    # Objects
    m_views = re.search(r"Views:\s+(\d+)", out)
    if m_views:
        metrics["views"] = int(m_views.group(1))
    m_act = re.search(r"Activities:\s+(\d+)", out)
    if m_act:
        metrics["activities"] = int(m_act.group(1))
    m_assets = re.search(r"Assets:\s+(\d+)", out)
    if m_assets:
        metrics["assets"] = int(m_assets.group(1))

    # Native Allocations
    m_bmp = re.search(r"Bitmap \(malloced\):\s+(\d+)\s+(\d+)", out)
    if m_bmp:
        metrics["bitmap_count"] = int(m_bmp.group(1))
        metrics["bitmap_kb"] = int(m_bmp.group(2))

    m_other = re.search(r"Other \(malloced\):\s+(\d+)\s+(\d+)", out)
    if m_other:
        metrics["other_malloc_count"] = int(m_other.group(1))
        metrics["other_malloc_kb"] = int(m_other.group(2))

    return metrics

def get_gc_logs():
    logs = run_adb(["logcat", "-d", "-s", "echogesture.app:I"])
    gc_entries = []
    for line in logs.splitlines():
        if "GC freed" in line:
            gc_entries.append(line)
    return gc_entries

def main():
    pid = get_pid()
    if not pid:
        print("Starting app...")
        run_adb(["shell", "am", "start", "-n", f"{PKG}/.ui.MainActivity"])
        time.sleep(3)
        pid = get_pid()
    
    print(f"Monitoring PID: {pid} on device {DEVICE}", flush=True)
    samples = []
    start_time = time.time()
    
    total_samples = 15
    for i in range(1, total_samples + 1):
        elapsed = time.time() - start_time
        current_pid = get_pid()
        if not current_pid:
            print(f"Sample {i}: App died or was killed!", flush=True)
            break
            
        phase_label = "STEADY_STATE_LIVE"
        
        # At sample 9, trigger Home (Pause / Stop)
        if i == 9:
            print(f"\n>>> [LIFECYCLE TRIGGER: onPause / onStop] Sending KEYCODE_HOME...", flush=True)
            run_adb(["shell", "input", "keyevent", "KEYCODE_HOME"])
            time.sleep(2)
            phase_label = "BACKGROUND_PAUSED"
        elif i == 10:
            phase_label = "BACKGROUND_PAUSED"
        # At sample 11, resume app
        elif i == 11:
            print(f"\n>>> [LIFECYCLE TRIGGER: onResume / Restart] Launching MainActivity...", flush=True)
            run_adb(["shell", "am", "start", "-n", f"{PKG}/.ui.MainActivity"])
            time.sleep(2)
            phase_label = "RESUMED_ACTIVE"
        elif i >= 12:
            phase_label = "RESUMED_STABILIZED"
            
        m = parse_meminfo(current_pid)
        m["sample_idx"] = i
        m["elapsed_sec"] = round(elapsed, 1)
        m["phase"] = phase_label
        
        print(f"Sample {i:02d} [{m['elapsed_sec']:>5.1f}s] ({phase_label:<20}): "
              f"PSS Total={m['pss_total']/1024:>6.1f}MB | Native={m['pss_native']/1024:>6.1f}MB | "
              f"Dalvik={m['pss_dalvik']/1024:>5.1f}MB | PrivDirty={m['private_dirty_total']/1024:>6.1f}MB | "
              f"Bitmaps={m['bitmap_count']:>2d} ({m['bitmap_kb']:>5d}KB) | Views={m['views']:>2d}", flush=True)
        
        clean_sample = {k: v for k, v in m.items() if k != "raw_output"}
        samples.append(clean_sample)
        
        if i < total_samples:
            time.sleep(8)
            
    gc_logs = get_gc_logs()
    
    results = {
        "device": DEVICE,
        "package": PKG,
        "pid": pid,
        "total_duration_sec": round(time.time() - start_time, 1),
        "sample_count": len(samples),
        "samples": samples,
        "recent_gc_events": gc_logs[-15:] if gc_logs else []
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/sustained_memory_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full audit results to reports/sustained_memory_audit_results.json", flush=True)

if __name__ == "__main__":
    main()
