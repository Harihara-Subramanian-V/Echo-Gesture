"""
EchoGesture Production Inference Runtime Wrapper
Lightweight, reproducible edge runtime for INT8 quantized sign language recognition.

Exposes:
- EchoGestureRuntime class
- predict(features: np.ndarray) -> dict

Contract:
- Input: np.ndarray of shape (64, 263) or (1, 64, 263), float32
- Output: Structured dict with predicted class, confidence, top-5 rankings, and latency
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np


class EchoGestureRuntime:
    """
    Production-ready runtime wrapper for EchoGesture INT8 TFLite model.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        num_threads: int = 1
    ):
        # Resolve default paths relative to repository root
        repo_root = Path(__file__).resolve().parent.parent
        self.model_path = model_path or str(repo_root / "models" / "quantized" / "echogesture_exp_d_int8.tflite")
        self.label_encoder_path = label_encoder_path or str(repo_root / "data" / "processed" / "sequences_v3" / "label_encoder.json")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Quantized TFLite model not found at: {self.model_path}")
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found at: {self.label_encoder_path}")
            
        # Load vocabulary mapping
        with open(self.label_encoder_path, "r", encoding="utf-8") as f:
            enc_data = json.load(f)
        self.idx2label = {int(k): v for k, v in enc_data["idx2label"].items()}
        self.num_classes = enc_data["num_classes"]
        
        # Initialize TFLite interpreter
        self._init_interpreter(num_threads)
        
    def _init_interpreter(self, num_threads: int):
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=num_threads)
        except ImportError:
            import tensorflow.lite as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path, num_threads=num_threads)
            
        self.interpreter.allocate_tensors()
        
        in_details = self.interpreter.get_input_details()[0]
        out_details = self.interpreter.get_output_details()[0]
        
        self.in_index = in_details["index"]
        self.out_index = out_details["index"]
        
        self.expected_shape = tuple(in_details["shape"])  # e.g. (1, 64, 263)
        self.in_scale, self.in_zero_point = in_details["quantization"]
        self.out_scale, self.out_zero_point = out_details["quantization"]
        
        if in_details["dtype"] != np.int8:
            raise ValueError(f"Expected int8 input tensor, got {in_details['dtype']}")
        if out_details["dtype"] != np.int8:
            raise ValueError(f"Expected int8 output tensor, got {out_details['dtype']}")

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run INT8 inference on a single preprocessed 263-D feature sequence.
        
        Args:
            features: np.ndarray of shape (64, 263) or (1, 64, 263), float32.
            
        Returns:
            dict containing:
                - predicted_class (str): Name of recognized sign
                - predicted_class_id (int): Integer index (0..39)
                - confidence (float): Dequantized model confidence score in [0.0, 1.0]
                - top_5 (list): Top 5 candidate signs with confidences
                - latency_ms (float): Model invoke time in milliseconds
        """
        # 1. Input Validation
        if not isinstance(features, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(features)}")
            
        if features.shape == (64, 263):
            feat_batch = np.expand_dims(features, axis=0)
        elif features.shape == (1, 64, 263):
            feat_batch = features
        else:
            raise ValueError(f"Invalid input shape {features.shape}. Expected (64, 263) or (1, 64, 263).")
            
        if np.isnan(feat_batch).any() or np.isinf(feat_batch).any():
            raise ValueError("Input features contain NaN or Inf values.")
            
        # 2. Input Quantization: q = clip(round(x / S) + Z, -128, 127)
        q_in = np.clip(
            np.round(feat_batch / self.in_scale) + self.in_zero_point,
            -128, 127
        ).astype(np.int8)
        
        # 3. Model Execution
        self.interpreter.set_tensor(self.in_index, q_in)
        t_start = time.perf_counter()
        self.interpreter.invoke()
        invoke_time_ms = (time.perf_counter() - t_start) * 1000.0
        
        # 4. Output Dequantization
        q_out = self.interpreter.get_tensor(self.out_index)[0]  # shape (40,) int8
        
        # Monotonic argmax: argmax of q_out is identical to argmax of probabilities
        pred_idx = int(np.argmax(q_out))
        pred_class = self.idx2label[pred_idx]
        
        # Dequantize logits to probabilities in [0.0, 1.0]
        # p = (q - Z_out) * S_out
        dequant_probs = (q_out.astype(np.float32) - float(self.out_zero_point)) * float(self.out_scale)
        pred_confidence = float(np.clip(dequant_probs[pred_idx], 0.0, 1.0))
        
        # Top-5 Ranking
        top5_indices = np.argsort(q_out)[-5:][::-1]
        top_5_list = []
        for rank, c_idx in enumerate(top5_indices, start=1):
            c_idx = int(c_idx)
            top_5_list.append({
                "rank": rank,
                "class_id": c_idx,
                "class_name": self.idx2label[c_idx],
                "confidence": float(np.clip(dequant_probs[c_idx], 0.0, 1.0))
            })
            
        return {
            "predicted_class": pred_class,
            "predicted_class_id": pred_idx,
            "confidence": pred_confidence,
            "top_5": top_5_list,
            "latency_ms": float(invoke_time_ms)
        }


# Global cached runtime instance for convenience
_cached_runtime: Optional[EchoGestureRuntime] = None


def predict(features: np.ndarray, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function matching the interface contract:
    predict(features: numpy.ndarray) -> dict
    """
    global _cached_runtime
    if _cached_runtime is None or (model_path and _cached_runtime.model_path != model_path):
        _cached_runtime = EchoGestureRuntime(model_path=model_path)
    return _cached_runtime.predict(features)


if __name__ == "__main__":
    print("Testing EchoGesture Runtime...")
    runtime = EchoGestureRuntime()
    print(f"Loaded model: {runtime.model_path}")
    print(f"Classes: {runtime.num_classes}")
    
    # Test on dummy input
    dummy_input = np.zeros((64, 263), dtype=np.float32)
    res = runtime.predict(dummy_input)
    print("\nDummy prediction result:")
    print(json.dumps(res, indent=2))
    assert "predicted_class" in res
    assert "top_5" in res
    assert len(res["top_5"]) == 5
    print("\nEchoGesture Runtime self-test PASSED!")
