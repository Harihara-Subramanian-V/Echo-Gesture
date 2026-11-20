import subprocess
import time
import re
import json
import os
import sys
from pathlib import Path

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
        "pss_total_kb": 0,
        "rss_total_kb": 0,
        "private_dirty_total_kb": 0,
        "pss_native_kb": 0,
        "heap_size_native_kb": 0,
        "heap_alloc_native_kb": 0,
        "heap_free_native_kb": 0,
        "pss_dalvik_kb": 0,
        "heap_size_dalvik_kb": 0,
        "heap_alloc_dalvik_kb": 0,
        "heap_free_dalvik_kb": 0,
        "dalvik_free_percent": 0.0,
        "graphics_kb": 0,
        "views": 0,
        "activities": 0,
        "bitmap_count": 0,
        "bitmap_kb": 0,
        "raw_output": out
    }
    
    # Native Heap
    m_native = re.search(r"Native Heap\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if m_native:
        metrics["pss_native_kb"] = int(m_native.group(1))
        metrics["heap_size_native_kb"] = int(m_native.group(3))
        metrics["heap_alloc_native_kb"] = int(m_native.group(4))
        metrics["heap_free_native_kb"] = int(m_native.group(5))

    # Dalvik Heap
    m_dalvik = re.search(r"Dalvik Heap\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)", out)
    if m_dalvik:
        metrics["pss_dalvik_kb"] = int(m_dalvik.group(1))
        metrics["heap_size_dalvik_kb"] = int(m_dalvik.group(3))
        metrics["heap_alloc_dalvik_kb"] = int(m_dalvik.group(4))
        metrics["heap_free_dalvik_kb"] = int(m_dalvik.group(5))
        if metrics["heap_size_dalvik_kb"] > 0:
            metrics["dalvik_free_percent"] = round((metrics["heap_free_dalvik_kb"] / metrics["heap_size_dalvik_kb"]) * 100, 1)

    # TOTAL PSS & RSS
    m_tot = re.search(r"TOTAL\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+(\d+)", out)
    if m_tot:
        metrics["pss_total_kb"] = int(m_tot.group(1))
        metrics["private_dirty_total_kb"] = int(m_tot.group(2))
        metrics["rss_total_kb"] = int(m_tot.group(3))

    # Graphics
    m_gfx = re.search(r"Graphics:\s+(\d+)", out)
    if m_gfx:
        metrics["graphics_kb"] = int(m_gfx.group(1))

    # Objects
    m_views = re.search(r"Views:\s+(\d+)", out)
    if m_views:
        metrics["views"] = int(m_views.group(1))
    m_act = re.search(r"Activities:\s+(\d+)", out)
    if m_act:
        metrics["activities"] = int(m_act.group(1))

    # Bitmaps
    m_bmp = re.search(r"Bitmap \(malloced\):\s+(\d+)\s+(\d+)", out)
    if m_bmp:
        metrics["bitmap_count"] = int(m_bmp.group(1))
        metrics["bitmap_kb"] = int(m_bmp.group(2))

    return metrics

def get_gc_events(pid):
    logs = run_adb(["shell", "logcat", "-d", "--pid", str(pid)])
    gc_entries = []
    for line in logs.splitlines():
        if any(keyword in line for keyword in ["GC", "CollectorTypeCC", "paused"]):
            gc_entries.append(line)
    return gc_entries

def main():
    pid = get_pid()
    if not pid:
        print("Starting app...")
        run_adb(["shell", "am", "start", "-n", f"{PKG}/.ui.MainActivity"])
        time.sleep(3)
        pid = get_pid()
    
    print(f"Monitoring sustained stability on PID: {pid} (vivo V2416 / Android 16)...", flush=True)
    samples = []
    start_time = time.time()
    
    total_snapshots = 12
    for i in range(1, total_snapshots + 1):
        elapsed = time.time() - start_time
        current_pid = get_pid()
        if not current_pid:
            print(f"Sample {i}: App died or was killed!", flush=True)
            break
            
        phase_label = "LIVE_OPTICAL_STREAM"
        
        # Lifecycle perturbation at sample 8 and 10
        if i == 8:
            print(f">>> [Snapshot 08] Lifecycle test: Sending KEYCODE_HOME (onPause/onStop)...", flush=True)
            run_adb(["shell", "input", "keyevent", "KEYCODE_HOME"])
            time.sleep(2)
            phase_label = "BACKGROUND_PAUSED"
        elif i == 9:
            phase_label = "BACKGROUND_PAUSED"
        elif i == 10:
            print(f">>> [Snapshot 10] Lifecycle resume: Re-launching MainActivity (onResume)...", flush=True)
            run_adb(["shell", "am", "start", "-n", f"{PKG}/.ui.MainActivity"])
            time.sleep(2)
            phase_label = "RESUMED_ACTIVE"
        elif i >= 11:
            phase_label = "RESUMED_STABILIZED"
            
        m = parse_meminfo(current_pid)
        m["snapshot_idx"] = i
        m["elapsed_sec"] = round(elapsed, 1)
        m["phase"] = phase_label
        
        pss_mb = m["pss_total_kb"] / 1024.0
        rss_mb = m["rss_total_kb"] / 1024.0
        native_mb = m["pss_native_kb"] / 1024.0
        dalvik_mb = m["pss_dalvik_kb"] / 1024.0
        dalvik_alloc_mb = m["heap_alloc_dalvik_kb"] / 1024.0
        dalvik_size_mb = m["heap_size_dalvik_kb"] / 1024.0
        gfx_mb = m["graphics_kb"] / 1024.0
        
        print(f"Snapshot {i:02d} [{m['elapsed_sec']:>5.1f}s] ({phase_label:<20}): "
              f"PSS={pss_mb:>5.1f}MB | RSS={rss_mb:>5.1f}MB | Native={native_mb:>5.1f}MB | "
              f"Dalvik={dalvik_mb:>4.1f}MB (Alloc={dalvik_alloc_mb:>4.1f}MB/{dalvik_size_mb:>4.1f}MB, {m['dalvik_free_percent']}% free) | "
              f"Gfx={gfx_mb:>4.1f}MB | Views={m['views']:>2d} | Act={m['activities']:>1d}", flush=True)
        
        clean_sample = {k: v for k, v in m.items() if k != "raw_output"}
        samples.append(clean_sample)
        
        if i < total_snapshots:
            time.sleep(8)
            
    gc_logs = get_gc_events(pid)
    
    results = {
        "device": DEVICE,
        "platform": "vivo V2416 (Android 16, API 36, arm64-v8a)",
        "package": PKG,
        "pid": pid,
        "total_snapshots": len(samples),
        "total_duration_sec": round(time.time() - start_time, 1),
        "samples": samples,
        "gc_events": gc_logs
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase7e_memory_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(samples)} memory snapshots to reports/phase7e_memory_audit_results.json", flush=True)

if __name__ == "__main__":
    main()
