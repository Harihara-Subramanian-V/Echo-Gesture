"""
EchoGesture: Official Track 02 Prediction Interface
Lightweight, reproducible edge runtime for sign language recognition.

Exposes:
- predict(video_path: str) -> dict
- EchoGesturePredictor
- OFFICIAL_VOCABULARY

Inference Contract:
- Input: Path to video file (str or Path)
- Output: Structured dictionary:
  {
      "clip_id": "<clip_id>",
      "prediction": "<UPPERCASE_GLOSS>",
      "top5": ["<GLOSS1>", "<GLOSS2>", "<GLOSS3>", "<GLOSS4>", "<GLOSS5>"],
      "confidence": float (in [0.0, 1.0]),
      "latency_ms": float (> 0.0)
  }
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Suppress verbose C++/absl/TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.live_inference import LiveLandmarkExtractor, LiveTemporalPreprocessor
from scripts.prepare_sequences import resample_sequence
from scripts.echogesture_runtime import EchoGestureRuntime


OFFICIAL_VOCABULARY = (
    "ALL", "BED", "BEFORE", "BLACK", "BLUE", "BOOK", "BOWLING", "CAN", "CANDY", "CHAIR",
    "CLOTHES", "COMPUTER", "COOL", "COUSIN", "DEAF", "DOG", "DRINK", "FAMILY", "FINE", "FINISH",
    "FISH", "GO", "GRADUATE", "HELP", "HOT", "LIKE", "MANY", "MOTHER", "NO", "NOW",
    "ORANGE", "TABLE", "THANKSGIVING", "THIN", "WALK", "WHAT", "WHO", "WOMAN", "YEAR", "YES"
)

VALID_PREDICTIONS = set(OFFICIAL_VOCABULARY) | {"IDLE"}

def _find_default(candidates: List[Path]) -> str:
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])

DEFAULT_MODEL_PATH = _find_default([
    REPO_ROOT / "model" / "echogesture_exp_d_int8.tflite",
    REPO_ROOT / "models" / "quantized" / "echogesture_exp_d_int8.tflite",
    REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "echogesture_exp_d_int8.tflite"
])
DEFAULT_SCALER_PATH = _find_default([
    REPO_ROOT / "model" / "scaler_params.npz",
    REPO_ROOT / "data" / "processed" / "sequences_v3" / "scaler_params.npz"
])
DEFAULT_LABEL_PATH = _find_default([
    REPO_ROOT / "model" / "label_encoder.json",
    REPO_ROOT / "data" / "processed" / "sequences_v3" / "label_encoder.json",
    REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "label_encoder.json"
])


class EchoGesturePredictor:
    """
    Stateful predictor wrapping MediaPipe Landmark Extractor,
    Temporal Resampler, Scaler/Velocity Preprocessor, and INT8 TFLite Runtime.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_threads: int = 1
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.scaler_path = scaler_path or DEFAULT_SCALER_PATH
        self.label_encoder_path = label_encoder_path or DEFAULT_LABEL_PATH

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler parameters not found at: {self.scaler_path}")
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found at: {self.label_encoder_path}")

        # Initialize pipeline components
        self.extractor = LiveLandmarkExtractor(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.preprocessor = LiveTemporalPreprocessor(scaler_path=self.scaler_path)
        self.runtime = EchoGestureRuntime(
            model_path=self.model_path,
            label_encoder_path=self.label_encoder_path,
            num_threads=num_threads
        )

        # Cache vocabulary mapping in uppercase
        with open(self.label_encoder_path, "r", encoding="utf-8") as f:
            enc_data = json.load(f)
        self.idx2label = {int(k): str(v).upper() for k, v in enc_data["idx2label"].items()}
        self.num_classes = enc_data["num_classes"]

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Execute end-to-end inference on a video file.

        Args:
            video_path: Path to video file (str or Path)

        Returns:
            Dict containing:
                - clip_id: str
                - prediction: str (uppercase gloss or "IDLE")
                - top5: List[str] (5 uppercase glosses)
                - confidence: float (in [0.0, 1.0])
                - latency_ms: float (> 0.0)
        """
        p = Path(video_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        clip_id = p.stem
        t_start = time.perf_counter()

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"Empty or corrupted video file (cannot open or decode): {video_path}")

        # Reset extractor temporal tracking for each video
        self.extractor.reset_tracking()
        if hasattr(self.extractor.detector, "reset"):
            self.extractor.detector.reset()

        frames_134: List[np.ndarray] = []
        any_hands_detected = False

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            flat_134, _, _ = self.extractor.process_frame(frame)
            frames_134.append(flat_134)

            # Check presence (Left hand index 126, Right hand index 127)
            if flat_134[126] > 0.5 or flat_134[127] > 0.5:
                any_hands_detected = True

        cap.release()

        # Handle empty or corrupted video (0 frames read)
        if len(frames_134) == 0:
            raise ValueError(f"Empty or corrupted video file (0 valid frames read): {video_path}")

        # Handle zero-hands video fallback
        if not any_hands_detected:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "clip_id": clip_id,
                "prediction": "IDLE",
                "top5": ["IDLE"] * 5,
                "confidence": 0.0,
                "latency_ms": round(float(latency_ms), 2)
            }

        # Temporal Resampling to fixed length L=64
        seq_134 = np.array(frames_134, dtype=np.float32)
        resampled_134 = resample_sequence(seq_134, target_len=64)

        # Preprocessing: Z-score standardization and 129-D velocity features -> 263-D
        features_263 = self.preprocessor.process_window(resampled_134)

        # Model inference via INT8 TFLite runtime
        raw_res = self.runtime.predict(features_263)

        top_class = str(raw_res["predicted_class"]).upper()
        top5_classes = [str(item["class_name"]).upper() for item in raw_res["top_5"]]
        confidence = float(raw_res["confidence"])

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "clip_id": clip_id,
            "prediction": top_class,
            "top5": top5_classes,
            "confidence": round(float(confidence), 4),
            "latency_ms": round(float(latency_ms), 2)
        }

    def close(self):
        """Release underlying detector and interpreter resources."""
        if hasattr(self, "extractor") and self.extractor:
            self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_cached_predictor: Optional[EchoGesturePredictor] = None


def get_default_predictor(
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    label_encoder_path: Optional[str] = None
) -> EchoGesturePredictor:
    """Retrieve or initialize the globally cached EchoGesturePredictor instance."""
    global _cached_predictor
    if _cached_predictor is None:
        _cached_predictor = EchoGesturePredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            label_encoder_path=label_encoder_path
        )
    return _cached_predictor


def predict(
    video_path: Union[str, Path],
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    label_encoder_path: Optional[str] = None,
    predictor: Optional[EchoGesturePredictor] = None
) -> Dict[str, Any]:
    """
    Official Track 02 prediction interface for EchoGesture.

    Args:
        video_path: Path to video file (str or Path).
        model_path: Optional custom path to INT8 TFLite model.
        scaler_path: Optional custom path to scaler_params.npz.
        label_encoder_path: Optional custom path to label_encoder.json.
        predictor: Optional custom EchoGesturePredictor instance.

    Returns:
        Structured prediction dictionary matching:
        {
            "clip_id": "<clip_id>",
            "prediction": "<UPPERCASE_GLOSS>",
            "top5": ["<GLOSS1>", "<GLOSS2>", "<GLOSS3>", "<GLOSS4>", "<GLOSS5>"],
            "confidence": 0.xx,
            "latency_ms": xx.x
        }
    """
    # Fast path validation: fail immediately if video file does not exist
    p_path = Path(video_path)
    if not p_path.exists() or not p_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if predictor is not None:
        return predictor.predict(video_path)

    global _cached_predictor
    if (
        _cached_predictor is not None and (
            (model_path and _cached_predictor.model_path != model_path) or
            (scaler_path and _cached_predictor.scaler_path != scaler_path) or
            (label_encoder_path and _cached_predictor.label_encoder_path != label_encoder_path)
        )
    ):
        _cached_predictor.close()
        _cached_predictor = None

    p = get_default_predictor(
        model_path=model_path,
        scaler_path=scaler_path,
        label_encoder_path=label_encoder_path
    )
    return p.predict(video_path)


def parse_args():
    parser = argparse.ArgumentParser(description="EchoGesture Track 02 Official Prediction Interface")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input video file (e.g. .mp4)")
    parser.add_argument("--model", type=str, default=None, help="Optional path to INT8 TFLite model")
    parser.add_argument("--scaler", type=str, default=None, help="Optional path to scaler_params.npz")
    parser.add_argument("--label_encoder", type=str, default=None, help="Optional path to label_encoder.json")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = predict(
            video_path=args.video,
            model_path=args.model,
            scaler_path=args.scaler,
            label_encoder_path=args.label_encoder
        )
        print(json.dumps(result, indent=2))
        return 0
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())
