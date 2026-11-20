import subprocess
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any

ADB_PATH = r"C:\Users\haris\AppData\Local\Android\Sdk\platform-tools\adb.exe"

def get_logcat_output() -> str:
    res = subprocess.run([ADB_PATH, "logcat", "-d"], capture_output=True, text=True, errors="ignore")
    return res.stdout

def parse_logs(log_text: str):
    audit_pattern = re.compile(
        r"AUDIT_FRAME:\s+frame=(\d+)\s+convMs=([\d\.]+)\s+rawDetectMs=([\d\.]+)\s+"
        r"detectMs=([\d\.]+)\s+featMs=([\d\.]+)\s+resampleMs=([\d\.]+)\s+prepMs=([\d\.]+)\s+"
        r"quantMs=([\d\.]+)\s+modelMs=([\d\.]+)\s+decMs=([\d\.]+)\s+infMs=([\d\.]+)\s+"
        r"stabMs=([\d\.]+)\s+totalMs=([\d\.]+)\s+fps=([\d\.]+)"
    )
    bench_pattern = re.compile(
        r"BENCH_STAGE:\s+frame=(\d+)\s+featMs=([\d\.]+)\s+resampleMs=([\d\.]+)\s+"
        r"prepMs=([\d\.]+)\s+quantMs=([\d\.]+)\s+modelMs=([\d\.]+)\s+decMs=([\d\.]+)\s+"
        r"infMs=([\d\.]+)\s+stabMs=([\d\.]+)\s+totalMs=([\d\.]+)"
    )
    vision_pattern = re.compile(
        r"BENCH_VISION:\s+iter=(\d+)\s+convMs=([\d\.]+)\s+rawDetectMs=([\d\.]+)\s+totalMs=([\d\.]+)"
    )
    preview_pattern = re.compile(r"previewFps=([\d\.]+)")

    audit_records = []
    bench_records = []
    vision_records = []
    preview_fps = []

    for line in log_text.splitlines():
        m_audit = audit_pattern.search(line)
        if m_audit:
            audit_records.append({
                "frame": int(m_audit.group(1)),
                "convMs": float(m_audit.group(2)),
                "rawDetectMs": float(m_audit.group(3)),
                "detectMs": float(m_audit.group(4)),
                "featMs": float(m_audit.group(5)),
                "resampleMs": float(m_audit.group(6)),
                "prepMs": float(m_audit.group(7)),
                "quantMs": float(m_audit.group(8)),
                "modelMs": float(m_audit.group(9)),
                "decMs": float(m_audit.group(10)),
                "infMs": float(m_audit.group(11)),
                "stabMs": float(m_audit.group(12)),
                "totalMs": float(m_audit.group(13)),
                "fps": float(m_audit.group(14))
            })
            continue

        m_bench = bench_pattern.search(line)
        if m_bench:
            bench_records.append({
                "frame": int(m_bench.group(1)),
                "featMs": float(m_bench.group(2)),
                "resampleMs": float(m_bench.group(3)),
                "prepMs": float(m_bench.group(4)),
                "quantMs": float(m_bench.group(5)),
                "modelMs": float(m_bench.group(6)),
                "decMs": float(m_bench.group(7)),
                "infMs": float(m_bench.group(8)),
                "stabMs": float(m_bench.group(9)),
                "totalMs": float(m_bench.group(10))
            })
            continue

        m_vision = vision_pattern.search(line)
        if m_vision:
            vision_records.append({
                "iter": int(m_vision.group(1)),
                "convMs": float(m_vision.group(2)),
                "rawDetectMs": float(m_vision.group(3)),
                "totalMs": float(m_vision.group(4))
            })
            continue

        m_fps = preview_pattern.search(line)
        if m_fps:
            preview_fps.append(float(m_fps.group(1)))

    return pd.DataFrame(audit_records), pd.DataFrame(bench_records), pd.DataFrame(vision_records), preview_fps

def compute_stats(series: pd.Series) -> Dict[str, float]:
    arr = series.dropna().values
    if len(arr) == 0:
        return {}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr))
    }

def print_stats_table(name: str, df: pd.DataFrame, columns: List[str]):
    print(f"\n==========================================================================================")
    print(f"  STATISTICS: {name} (N = {len(df)})")
    print(f"==========================================================================================")
    header = f"{'Stage':<14} | {'Mean':<7} | {'Median':<7} | {'P90':<7} | {'P95':<7} | {'P99':<7} | {'Max':<7} | {'Min':<7} | {'Std':<7}"
    print(header)
    print("-" * len(header))
    for col in columns:
        if col not in df:
            continue
        st = compute_stats(df[col])
        print(f"{col:<14} | {st['mean']:<7.2f} | {st['median']:<7.2f} | {st['p90']:<7.2f} | {st['p95']:<7.2f} | {st['p99']:<7.2f} | {st['max']:<7.2f} | {st['min']:<7.2f} | {st['std']:<7.2f}")

def main():
    log_text = get_logcat_output()
    df_audit, df_bench, df_vision, preview_fps = parse_logs(log_text)

    print(f"Loaded {len(df_audit)} Live Optical Frames")
    print(f"Loaded {len(df_bench)} Full Pipeline Benchmark Frames")
    print(f"Loaded {len(df_vision)} Isolated Vision Benchmark Iterations")
    print(f"Loaded {len(preview_fps)} Preview FPS Measurements: Mean = {np.mean(preview_fps):.2f} FPS")

    # Save cleanly to disk
    df_audit.to_csv("logs/audit_frames.csv", index=False)
    df_bench.to_csv("logs/bench_stage.csv", index=False)
    df_vision.to_csv("logs/bench_vision.csv", index=False)

    # 1. Live Optical Capture & Vision Stats
    if len(df_audit) > 0:
        print_stats_table(
            "Live CameraX Optical Detection Loop (Physical Hardware)",
            df_audit,
            ["convMs", "rawDetectMs", "detectMs", "featMs", "totalMs", "fps"]
        )

    # 2. Stage-Wise Recognition & Inference Stats (Warm frames >= frame 16)
    if len(df_bench) > 0:
        df_bench_active = df_bench[df_bench["frame"] >= 16]
        print_stats_table(
            "Stage-Wise Inference & Recognition Loop (N = 485 Active Inference Frames)",
            df_bench_active,
            ["featMs", "resampleMs", "prepMs", "quantMs", "modelMs", "decMs", "infMs", "stabMs", "totalMs"]
        )

    # 3. Vision Benchmark
    if len(df_vision) > 0:
        print_stats_table(
            "Isolated MediaPipe Vision Engine (640x480 Bitmap)",
            df_vision,
            ["convMs", "rawDetectMs", "totalMs"]
        )

if __name__ == "__main__":
    main()
