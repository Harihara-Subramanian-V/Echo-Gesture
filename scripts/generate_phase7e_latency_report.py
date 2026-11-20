import pandas as pd
import numpy as np
import os
from pathlib import Path

def compute_stats(series: pd.Series) -> dict:
    arr = series.dropna().values
    if len(arr) == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }

def main():
    repo_root = Path(__file__).resolve().parent.parent
    logs_dir = repo_root / "logs"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    df_audit = pd.read_csv(logs_dir / "audit_frames.csv")
    df_bench = pd.read_csv(logs_dir / "bench_stage.csv")
    df_vision = pd.read_csv(logs_dir / "bench_vision.csv")

    # Filter warm frames for bench (>= 16 frames after buffer fills)
    df_bench_active = df_bench[df_bench["frame"] >= 16]

    print(f"Loaded {len(df_audit)} live optical frames from audit_frames.csv")
    print(f"Loaded {len(df_bench_active)} active inference frames from bench_stage.csv (total: {len(df_bench)})")
    print(f"Loaded {len(df_vision)} isolated vision benchmark iterations from bench_vision.csv")

    # Construct End-to-End full recognition cycle dataset
    # In live recognition, total cycle = convMs + rawDetectMs + featMs + resampleMs + prepMs + quantMs + modelMs + decMs + stabMs
    # Synthesize end-to-end full recognition latency distribution:
    min_len = min(len(df_audit), len(df_bench_active))
    audit_slice = df_audit.iloc[:min_len].reset_index(drop=True)
    bench_slice = df_bench_active.iloc[:min_len].reset_index(drop=True)

    e2e_series = (
        audit_slice["convMs"] +
        audit_slice["rawDetectMs"] +
        bench_slice["featMs"] +
        bench_slice["resampleMs"] +
        bench_slice["prepMs"] +
        bench_slice["quantMs"] +
        bench_slice["modelMs"] +
        bench_slice["decMs"] +
        bench_slice["stabMs"]
    )

    stage_definitions = [
        ("convMs", audit_slice["convMs"], "CameraX ImageProxy to Bitmap / MPImage format conversion"),
        ("rawDetectMs", audit_slice["rawDetectMs"], "MediaPipe HandLandmarker neural inference"),
        ("detectMs", audit_slice["detectMs"], "Total Detection Stage (convMs + rawDetectMs)"),
        ("featMs", bench_slice["featMs"], "Hungarian slot resolution & 134-D feature vector extraction"),
        ("resampleMs", bench_slice["resampleMs"], "LiveTemporalBuffer linear resampling to 64 frames"),
        ("prepMs", bench_slice["prepMs"], "263-D normalization & inter-frame velocity extraction"),
        ("quantMs", bench_slice["quantMs"], "Affine asymmetric INT8 quantization (16,832 floats)"),
        ("modelMs", bench_slice["modelMs"], "LiteRT INT8 neural model forward execution (Exp D)"),
        ("decMs", bench_slice["decMs"], "Top-5 softmax ranking & confidence dequantization"),
        ("infMs", bench_slice["infMs"], "Combined inference pipeline (prepMs + quantMs + modelMs + decMs)"),
        ("stabMs", bench_slice["stabMs"], "PredictionStabilizer hysteresis filter & state update"),
        ("totalMs_e2e", e2e_series, "End-to-end live recognition cycle (optical capture to HUD display)"),
        ("totalMs_audit", audit_slice["totalMs"], "Live optical frame capture cycle (steady state)"),
        ("totalMs_bench", bench_slice["totalMs"], "Stage-wise inference & recognition loop cycle"),
        ("fps", audit_slice["fps"], "Effective live recognition pipeline frame rate (FPS)")
    ]

    records = []
    for stage_name, series, description in stage_definitions:
        st = compute_stats(series)
        records.append({
            "stage": stage_name,
            "description": description,
            "sample_count": len(series),
            "mean_ms": round(st["mean"], 3),
            "median_ms": round(st["median"], 3),
            "p90_ms": round(st["p90"], 3),
            "p95_ms": round(st["p95"], 3),
            "p99_ms": round(st["p99"], 3),
            "min_ms": round(st["min"], 3),
            "max_ms": round(st["max"], 3),
            "std_ms": round(st["std"], 3)
        })

    report_df = pd.DataFrame(records)
    csv_path = reports_dir / "phase7e_latency_report.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"\nGenerated deliverable: {csv_path}")

    # Display clean formatted table
    print("\n" + "=" * 115)
    print(f"{'ECHOGESTURE PHASE 7E: PHYSICAL DEVICE NANOSECOND LATENCY BREAKDOWN':^115}")
    print(f"{'Hardware: vivo V2416 (Qualcomm Snapdragon arm64-v8a, Android 16 API 36)':^115}")
    print("=" * 115)
    header = f"{'Stage':<15} | {'Mean (ms)':<9} | {'Median':<8} | {'P90':<8} | {'P95':<8} | {'P99':<8} | {'Min':<7} | {'Max':<8} | {'Std':<7}"
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['stage']:<15} | {r['mean_ms']:<9.3f} | {r['median_ms']:<8.3f} | {r['p90_ms']:<8.3f} | {r['p95_ms']:<8.3f} | {r['p99_ms']:<8.3f} | {r['min_ms']:<7.3f} | {r['max_ms']:<8.3f} | {r['std_ms']:<7.3f}")
    print("=" * 115)

    # Decoupling summary
    model_st = compute_stats(bench_slice["modelMs"])
    e2e_st = compute_stats(e2e_series)
    ratio_mean = (model_st["mean"] / e2e_st["mean"]) * 100
    ratio_med = (model_st["median"] / e2e_st["median"]) * 100
    print(f"\n[DECOUPLING ANALYSIS]")
    print(f"• Pure LiteRT Neural Model Execution (modelMs): Mean = {model_st['mean']:.3f} ms | Median = {model_st['median']:.3f} ms | Min = {model_st['min']:.3f} ms")
    print(f"• Full Recognition Cycle Latency (totalMs_e2e):   Mean = {e2e_st['mean']:.3f} ms | Median = {e2e_st['median']:.3f} ms | P95 = {e2e_st['p95']:.3f} ms")
    print(f"• Pure Neural Model Proportion: {ratio_mean:.2f}% of mean total latency ({ratio_med:.2f}% of median)")
    print(f"• MediaPipe Hand Landmark Detection Proportion: {(compute_stats(audit_slice['detectMs'])['mean'] / e2e_st['mean']) * 100:.2f}% of mean total latency")

if __name__ == "__main__":
    main()
