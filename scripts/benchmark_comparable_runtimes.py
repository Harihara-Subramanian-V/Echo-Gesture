"""
EchoGesture Phase 5C: Comprehensive Rigorous Edge Runtime Benchmark
Evaluates across 1,000 timed iterations under strict experimental control:

Architectures & Runtimes:
1. Keras Float32 (best_improved_model_v5a.keras)
2. TFLite Float32 (echogesture_exp_d_float32.tflite)
3. TFLite Full INT8 (echogesture_exp_d_int8.tflite)

Methodology:
- 1,000 timed iterations preceded by 50 warm-up runs
- Single-thread (num_threads=1) baseline for hardware-neutral parity
- Decomposition into:
    a) Input Quantization & Preprocessing Latency
    b) Pure Model Invoke Latency
    c) Output Dequantization & Interpretation Latency
    d) End-to-End Pipeline Latency
- Statistically rigorous percentiles: Mean, Median, P90, P95, P99, Min, Max, Throughput
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from scripts.augment_sequences import compute_velocity_features


def benchmark_single_runtime(
    model_type: str,
    model_obj: Any,
    input_sample_float: np.ndarray,
    num_runs: int = 1000,
    num_warmup: int = 50
) -> Dict[str, Any]:
    """
    Measure model-only latency, input prep latency, and end-to-end latency over 1,000 runs.
    """
    if model_type == "keras_float32":
        model = model_obj
        k_in = np.expand_dims(input_sample_float.astype(np.float32), axis=0)
        
        # Warmup
        for _ in range(num_warmup):
            _ = model(k_in, training=False)
            
        invoke_timings = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(k_in, training=False)
            invoke_timings.append((time.perf_counter() - t0) * 1000.0)
            
        prep_timings = [0.0] * num_runs  # No quantization needed
        post_timings = [0.0] * num_runs
        
    elif model_type in ["tflite_float32", "tflite_int8"]:
        interp = model_obj
        in_details = interp.get_input_details()[0]
        out_details = interp.get_output_details()[0]
        in_idx = in_details["index"]
        out_idx = out_details["index"]
        is_int8 = (model_type == "tflite_int8")
        
        if is_int8:
            s_in, z_in = in_details["quantization"]
            s_out, z_out = out_details["quantization"]
            
        # 1. Warmup
        for _ in range(num_warmup):
            if is_int8:
                q_in = np.clip(np.round(input_sample_float / s_in) + z_in, -128, 127).astype(np.int8)
                interp.set_tensor(in_idx, np.expand_dims(q_in, axis=0))
            else:
                interp.set_tensor(in_idx, np.expand_dims(input_sample_float.astype(np.float32), axis=0))
            interp.invoke()
            _ = interp.get_tensor(out_idx)
            
        prep_timings = []
        invoke_timings = []
        post_timings = []
        
        # Set tensor buffer once for pure invoke isolation
        if is_int8:
            q_in = np.clip(np.round(input_sample_float / s_in) + z_in, -128, 127).astype(np.int8)
            interp.set_tensor(in_idx, np.expand_dims(q_in, axis=0))
        else:
            interp.set_tensor(in_idx, np.expand_dims(input_sample_float.astype(np.float32), axis=0))
            
        # Timed Pure Model Invoke
        for _ in range(num_runs):
            t0 = time.perf_counter()
            interp.invoke()
            invoke_timings.append((time.perf_counter() - t0) * 1000.0)
            
        # Timed Preprocessing / Quantization
        if is_int8:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                q = np.clip(np.round(input_sample_float / s_in) + z_in, -128, 127).astype(np.int8)
                interp.set_tensor(in_idx, np.expand_dims(q, axis=0))
                prep_timings.append((time.perf_counter() - t0) * 1000.0)
                
            for _ in range(num_runs):
                t0 = time.perf_counter()
                q_out = interp.get_tensor(out_idx)[0]
                _ = (q_out.astype(np.float32) - z_out) * s_out
                _ = int(np.argmax(q_out))
                post_timings.append((time.perf_counter() - t0) * 1000.0)
        else:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                interp.set_tensor(in_idx, np.expand_dims(input_sample_float.astype(np.float32), axis=0))
                prep_timings.append((time.perf_counter() - t0) * 1000.0)
            for _ in range(num_runs):
                t0 = time.perf_counter()
                out = interp.get_tensor(out_idx)[0]
                _ = int(np.argmax(out))
                post_timings.append((time.perf_counter() - t0) * 1000.0)
                
    invoke_arr = np.array(invoke_timings)
    prep_arr = np.array(prep_timings)
    post_arr = np.array(post_timings)
    e2e_arr = invoke_arr + prep_arr + post_arr
    
    return {
        "model_type": model_type,
        "iterations": num_runs,
        "model_only_ms": {
            "mean": float(np.mean(invoke_arr)),
            "std": float(np.std(invoke_arr)),
            "median": float(np.median(invoke_arr)),
            "p90": float(np.percentile(invoke_arr, 90)),
            "p95": float(np.percentile(invoke_arr, 95)),
            "p99": float(np.percentile(invoke_arr, 99)),
            "min": float(np.min(invoke_arr)),
            "max": float(np.max(invoke_arr)),
            "throughput_fps": float(1000.0 / np.mean(invoke_arr))
        },
        "preprocessing_ms": {
            "mean": float(np.mean(prep_arr)),
            "median": float(np.median(prep_arr))
        },
        "postprocessing_ms": {
            "mean": float(np.mean(post_arr)),
            "median": float(np.median(post_arr))
        },
        "end_to_end_ms": {
            "mean": float(np.mean(e2e_arr)),
            "median": float(np.median(e2e_arr)),
            "p95": float(np.percentile(e2e_arr, 95)),
            "throughput_fps": float(1000.0 / np.mean(e2e_arr))
        }
    }


def run_comprehensive_benchmark(
    keras_path: str = "models/checkpoints/best_improved_model_v5a.keras",
    f32_tflite_path: str = "models/quantized/echogesture_exp_d_float32.tflite",
    int8_tflite_path: str = "models/quantized/echogesture_exp_d_int8.tflite",
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports",
    num_runs: int = 1000
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"=== EchoGesture Phase 5C: Comprehensive Rigorous Benchmark ({num_runs} Iterations) ===")
    
    # 1. Load sample
    test_npz = np.load(os.path.join(data_dir, "test_sequences.npz"))
    sample_134 = test_npz["X"][0]
    sample_263 = compute_velocity_features(sample_134).astype(np.float32)
    
    # 2. Benchmark Keras Float32
    print(f"\n1. Benchmarking Keras Float32 ({num_runs} runs)...")
    k_model = keras.models.load_model(keras_path)
    res_keras = benchmark_single_runtime("keras_float32", k_model, sample_263, num_runs=num_runs, num_warmup=50)
    res_keras["file_size_bytes"] = os.path.getsize(keras_path)
    res_keras["file_size_kb"] = res_keras["file_size_bytes"] / 1024.0
    res_keras["file_size_mb"] = res_keras["file_size_bytes"] / (1024.0 * 1024.0)
    
    # 3. Benchmark TFLite Float32 (single-threaded parity)
    print(f"\n2. Benchmarking TFLite Float32 ({num_runs} runs, single thread)...")
    f32_interp = tf.lite.Interpreter(model_path=f32_tflite_path, num_threads=1)
    f32_interp.allocate_tensors()
    res_f32 = benchmark_single_runtime("tflite_float32", f32_interp, sample_263, num_runs=num_runs, num_warmup=50)
    res_f32["file_size_bytes"] = os.path.getsize(f32_tflite_path)
    res_f32["file_size_kb"] = res_f32["file_size_bytes"] / 1024.0
    res_f32["file_size_mb"] = res_f32["file_size_bytes"] / (1024.0 * 1024.0)
    
    # 4. Benchmark TFLite INT8 (single-threaded parity)
    print(f"\n3. Benchmarking TFLite INT8 ({num_runs} runs, single thread)...")
    int8_interp = tf.lite.Interpreter(model_path=int8_tflite_path, num_threads=1)
    int8_interp.allocate_tensors()
    res_int8 = benchmark_single_runtime("tflite_int8", int8_interp, sample_263, num_runs=num_runs, num_warmup=50)
    res_int8["file_size_bytes"] = os.path.getsize(int8_tflite_path)
    res_int8["file_size_kb"] = res_int8["file_size_bytes"] / 1024.0
    res_int8["file_size_mb"] = res_int8["file_size_bytes"] / (1024.0 * 1024.0)
    
    # Compile comparison
    all_results = {
        "keras_float32": res_keras,
        "tflite_float32": res_f32,
        "tflite_int8": res_int8,
        "analysis": {
            "framework_speedup_tflite_vs_keras": float(res_keras["model_only_ms"]["mean"] / res_f32["model_only_ms"]["mean"]),
            "quantization_speedup_int8_vs_f32": float(res_f32["model_only_ms"]["mean"] / res_int8["model_only_ms"]["mean"]),
            "total_speedup_int8_vs_keras": float(res_keras["model_only_ms"]["mean"] / res_int8["model_only_ms"]["mean"]),
            "model_size_compression_ratio": float(res_keras["file_size_bytes"] / res_int8["file_size_bytes"]),
            "quantization_overhead_pct_of_e2e": float(res_int8["preprocessing_ms"]["mean"] / res_int8["end_to_end_ms"]["mean"] * 100.0)
        }
    }
    
    json_path = out_path / "comparable_runtime_benchmarks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved comparable benchmark results to {json_path}")
    
    # Print formatted table
    print("\n=========================================================================================================")
    print("                     PHASE 5C RIGOROUS BENCHMARK RESULTS (1,000 TIMED ITERATIONS)                       ")
    print("=========================================================================================================")
    print(f" Runtime / Format   | Model Size | Model-Only Mean (P95) | Preproc / Quant | End-to-End Mean | Throughput")
    print(f" -------------------+------------+-----------------------+-----------------+-----------------+-----------")
    print(f" Keras Float32      | {res_keras['file_size_mb']:5.2f} MB   | {res_keras['model_only_ms']['mean']:6.3f} ms ({res_keras['model_only_ms']['p95']:6.3f} ms)|        N/A      | {res_keras['end_to_end_ms']['mean']:6.3f} ms    | {res_keras['model_only_ms']['throughput_fps']:7.1f} FPS")
    print(f" TFLite Float32     | {res_f32['file_size_kb']:5.1f} KB   | {res_f32['model_only_ms']['mean']:6.3f} ms ({res_f32['model_only_ms']['p95']:6.3f} ms)|     {res_f32['preprocessing_ms']['mean']:6.3f} ms   | {res_f32['end_to_end_ms']['mean']:6.3f} ms    | {res_f32['model_only_ms']['throughput_fps']:7.1f} FPS")
    print(f" TFLite INT8        | {res_int8['file_size_kb']:5.1f} KB   | {res_int8['model_only_ms']['mean']:6.3f} ms ({res_int8['model_only_ms']['p95']:6.3f} ms)|     {res_int8['preprocessing_ms']['mean']:6.3f} ms   | {res_int8['end_to_end_ms']['mean']:6.3f} ms    | {res_int8['model_only_ms']['throughput_fps']:7.1f} FPS")
    print("=========================================================================================================")
    print(f" Framework Optimization (Keras -> TFLite Float32):  {all_results['analysis']['framework_speedup_tflite_vs_keras']:.1f}x speedup")
    print(f" Quantization Optimization (TFLite Float32 -> INT8): {all_results['analysis']['quantization_speedup_int8_vs_f32']:.2f}x speedup")
    print(f" Total Memory Compression:                          {all_results['analysis']['model_size_compression_ratio']:.1f}x compression")
    print("=========================================================================================================")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Rigorous Comparable Benchmark")
    parser.add_argument("--num_runs", type=int, default=1000)
    args = parser.parse_args()
    
    run_comprehensive_benchmark(num_runs=args.num_runs)
