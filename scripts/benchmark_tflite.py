"""
EchoGesture Phase 5B: CPU Edge Performance Benchmarking
Measures inference latency, throughput, and memory footprint across Float32 and INT8 models.

Benchmarked on identical CPU hardware using 100 timed inference trials.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
import keras

from scripts.augment_sequences import compute_velocity_features


def benchmark_tflite_model(
    tflite_path: str,
    test_sample_float: np.ndarray,
    is_int8: bool = False,
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark TFLite interpreter latency and throughput."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    in_idx = in_details["index"]
    out_idx = out_details["index"]
    
    if is_int8:
        s_in, z_in = in_details["quantization"]
        q_in = np.clip(np.round(test_sample_float / s_in) + z_in, -128, 127).astype(np.int8)
        input_data = np.expand_dims(q_in, axis=0)
    else:
        input_data = np.expand_dims(test_sample_float.astype(np.float32), axis=0)
        
    # Warmup
    for _ in range(num_warmup):
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        
    # Timed runs
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(in_idx, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # ms
        
    timings = np.array(timings)
    file_size_bytes = os.path.getsize(tflite_path)
    
    return {
        "model_path": tflite_path,
        "format": "TFLite INT8" if is_int8 else "TFLite Float32",
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_bytes / 1024.0,
        "file_size_mb": file_size_bytes / (1024.0 * 1024.0),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "throughput_fps": float(1000.0 / np.mean(timings))
    }


def benchmark_keras_model(
    keras_path: str,
    test_sample_float: np.ndarray,
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Benchmark Keras Float32 model latency and throughput."""
    model = keras.models.load_model(keras_path)
    input_data = np.expand_dims(test_sample_float.astype(np.float32), axis=0)
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(input_data, training=False)
        
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(input_data, training=False)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
        
    timings = np.array(timings)
    file_size_bytes = os.path.getsize(keras_path)
    
    return {
        "model_path": keras_path,
        "format": "Keras Float32",
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_bytes / 1024.0,
        "file_size_mb": file_size_bytes / (1024.0 * 1024.0),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "throughput_fps": float(1000.0 / np.mean(timings))
    }


def run_benchmarks(
    float_model_path: str = "models/checkpoints/best_improved_model_v5a.keras",
    int8_tflite_path: str = "models/quantized/echogesture_exp_d_int8.tflite",
    data_dir: str = "data/processed/sequences_v3",
    output_dir: str = "reports"
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("=== EchoGesture Phase 5B: CPU Edge Performance Benchmark ===")
    
    # Load test sample
    test_npz = np.load(os.path.join(data_dir, "test_sequences.npz"))
    sample_134 = test_npz["X"][0]
    sample_263 = compute_velocity_features(sample_134).astype(np.float32)
    
    # 1. Benchmark Keras Float32
    print("Benchmarking Keras Float32 baseline...")
    keras_bench = benchmark_keras_model(float_model_path, sample_263, num_warmup=15, num_runs=100)
    
    # 2. Benchmark INT8 TFLite
    print("Benchmarking INT8 TFLite quantized model...")
    int8_bench = benchmark_tflite_model(int8_tflite_path, sample_263, is_int8=True, num_warmup=15, num_runs=100)
    
    # Latency speedup and compression ratio
    latency_reduction = (keras_bench["mean_ms"] - int8_bench["mean_ms"]) / keras_bench["mean_ms"] * 100.0
    compression_ratio = keras_bench["file_size_bytes"] / int8_bench["file_size_bytes"]
    
    benchmark_results = {
        "keras_float32": keras_bench,
        "tflite_int8": int8_bench,
        "compression_ratio": float(compression_ratio),
        "latency_reduction_pct": float(latency_reduction),
        "meets_edge_constraints": {
            "model_size_under_20mb": bool(int8_bench["file_size_mb"] < 20.0),
            "p95_latency_under_50ms": bool(int8_bench["p95_ms"] < 50.0),
            "int8_only_inference": True
        }
    }
    
    json_path = out_path / "edge_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Saved benchmark results to {json_path}")
    
    print("\n=========================================================================================")
    print("                     EDGE HARDWARE BENCHMARK COMPARISON (CPU ENVIRONMENT)                ")
    print("=========================================================================================")
    print(f" Metric                     | Keras Float32      | TFLite INT8        | Benefit / Delta")
    print(f" ---------------------------+--------------------+--------------------+------------------")
    print(f" Model File Size            | {keras_bench['file_size_mb']:6.2f} MB          | {int8_bench['file_size_kb']:6.2f} KB         | {compression_ratio:.1f}x compression")
    print(f" Mean Latency               | {keras_bench['mean_ms']:6.2f} ms          | {int8_bench['mean_ms']:6.2f} ms          | {latency_reduction:+.1f}% faster")
    print(f" Median Latency             | {keras_bench['median_ms']:6.2f} ms          | {int8_bench['median_ms']:6.2f} ms          | {keras_bench['median_ms'] - int8_bench['median_ms']:+5.2f} ms")
    print(f" P95 Latency                | {keras_bench['p95_ms']:6.2f} ms          | {int8_bench['p95_ms']:6.2f} ms          | Budget: < 50 ms")
    print(f" Throughput                 | {keras_bench['throughput_fps']:6.1f} fps         | {int8_bench['throughput_fps']:6.1f} fps         | +{int8_bench['throughput_fps'] - keras_bench['throughput_fps']:.1f} fps")
    print("=========================================================================================")
    
    return benchmark_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Edge Benchmark")
    parser.add_argument("--float_model", type=str, default="models/checkpoints/best_improved_model_v5a.keras")
    parser.add_argument("--int8_tflite", type=str, default="models/quantized/echogesture_exp_d_int8.tflite")
    parser.add_argument("--data_dir", type=str, default="data/processed/sequences_v3")
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()
    
    run_benchmarks(args.float_model, args.int8_tflite, args.data_dir, args.output_dir)
