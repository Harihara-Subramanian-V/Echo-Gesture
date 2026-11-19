"""
EchoGesture Phase 6A: Live Camera Inference Pipeline
Full real-time sign language recognition running on live webcam stream with INT8 TFLite runtime.

Pipeline Stages:
1. MediaPipe Hands tracking (Viewer-Left Slot 0 / Viewer-Right Slot 1)
2. 134-D frame feature construction
3. Rolling 64-frame temporal buffer
4. Active-frame Z-score standardization (scaler_params.npz)
5. First-order velocity extraction with transition spike prevention (263-D)
6. INT8 TFLite model inference (EchoGestureRuntime)
7. Temporal prediction stabilization & confidence smoothing
8. Real-time engineering HUD with per-stage latency breakdown
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

# Suppress verbose TensorFlow / absl logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mediapipe as mp
from scripts.echogesture_runtime import EchoGestureRuntime
from scripts.prepare_sequences import resample_sequence


class LiveLandmarkExtractor:
    """
    Online MediaPipe Hands feature extractor implementing the exact contract
    from scripts/extract_landmarks.py and reports/production_inference_contract.md.
    """
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.detector = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Temporal tracking cache for ambiguous multi-hand slot resolution
        self.prev_wrists: Optional[np.ndarray] = None
        self.prev_presence = np.zeros(2, dtype=np.float32)
        
    def process_frame(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        """
        Process a single BGR camera frame and extract the canonical 134-D feature vector.
        
        Returns:
            flat_134: (134,) float32 array
            telemetry: dict with detection flags, scores, and counts
            results: raw MediaPipe results for HUD visualization
        """
        h_px, w_px, _ = frame_bgr.shape
        frame_diagonal = float(np.sqrt(w_px**2 + h_px**2))
        
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.detector.process(rgb_frame)
        
        # Containers: Slot 0 = Viewer-Left Hand, Slot 1 = Viewer-Right Hand
        frame_norm = np.zeros((2, 21, 3), dtype=np.float32)
        frame_pres = np.zeros(2, dtype=np.float32)
        frame_h_score = np.zeros(2, dtype=np.float32)
        
        left_detected = False
        right_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            num_hands = len(results.multi_hand_landmarks)
            detected_hands = []
            
            for h_idx in range(num_hands):
                h_lms = results.multi_hand_landmarks[h_idx]
                handedness_info = results.multi_handedness[h_idx].classification[0]
                label = handedness_info.label  # 'Left' or 'Right'
                score = float(handedness_info.score)
                
                norm_coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
                detected_hands.append({
                    "label": label,
                    "score": score,
                    "norm": norm_coords,
                    "wrist_xy": norm_coords[0, :2]
                })
                
            if num_hands == 1:
                h = detected_hands[0]
                slot = 0 if h["label"] == "Left" else 1
                frame_norm[slot] = h["norm"]
                frame_pres[slot] = 1.0
                frame_h_score[slot] = h["score"]
                if slot == 0:
                    left_detected = True
                else:
                    right_detected = True
            elif num_hands >= 2:
                left_cands = [h for h in detected_hands if h["label"] == "Left"]
                right_cands = [h for h in detected_hands if h["label"] == "Right"]
                
                if left_cands and right_cands:
                    best_left = max(left_cands, key=lambda x: x["score"])
                    best_right = max(right_cands, key=lambda x: x["score"])
                    
                    frame_norm[0] = best_left["norm"]
                    frame_pres[0] = 1.0
                    frame_h_score[0] = best_left["score"]
                    left_detected = True
                    
                    frame_norm[1] = best_right["norm"]
                    frame_pres[1] = 1.0
                    frame_h_score[1] = best_right["score"]
                    right_detected = True
                else:
                    # Ambiguous duplicate labels: Hungarian tracking vs prev_wrists
                    c0, c1 = detected_hands[0], detected_hands[1]
                    if self.prev_wrists is not None and self.prev_presence[0] == 1.0 and self.prev_presence[1] == 1.0:
                        cost_direct = float(np.linalg.norm(c0["wrist_xy"] - self.prev_wrists[0]) + np.linalg.norm(c1["wrist_xy"] - self.prev_wrists[1]))
                        cost_swap = float(np.linalg.norm(c1["wrist_xy"] - self.prev_wrists[0]) + np.linalg.norm(c0["wrist_xy"] - self.prev_wrists[1]))
                        h_slot0, h_slot1 = (c1, c0) if cost_swap < cost_direct else (c0, c1)
                    else:
                        h_slot0, h_slot1 = (c0, c1) if c0["wrist_xy"][0] <= c1["wrist_xy"][0] else (c1, c0)
                        
                    frame_norm[0] = h_slot0["norm"]
                    frame_pres[0] = 1.0
                    frame_h_score[0] = h_slot0["score"]
                    left_detected = True
                    
                    frame_norm[1] = h_slot1["norm"]
                    frame_pres[1] = 1.0
                    frame_h_score[1] = h_slot1["score"]
                    right_detected = True
                    
        # Update tracking cache
        self.prev_wrists = np.array([frame_norm[0, 0, :2], frame_norm[1, 0, :2]], dtype=np.float32)
        self.prev_presence = frame_pres.copy()
        
        # Aspect-ratio corrected geometric inter-hand features
        if left_detected and right_detected:
            dx_norm = float(frame_norm[1, 0, 0] - frame_norm[0, 0, 0])
            dy_norm = float(frame_norm[1, 0, 1] - frame_norm[0, 0, 1])
            dx_px = dx_norm * w_px
            dy_px = dy_norm * h_px
            dist_px = float(np.sqrt(dx_px**2 + dy_px**2))
            inter_dist = float(dist_px / frame_diagonal)
            inter_offset = np.array([dx_px / frame_diagonal, dy_px / frame_diagonal], dtype=np.float32)
            both_present = 1.0
        else:
            inter_dist = 0.0
            inter_offset = np.zeros(2, dtype=np.float32)
            both_present = 0.0
            
        # Assemble exact 134-D vector
        flat_134 = np.concatenate([
            frame_norm[0].flatten(),                                     # 63
            frame_norm[1].flatten(),                                     # 63
            np.array([frame_pres[0]], dtype=np.float32),                 # 1
            np.array([frame_pres[1]], dtype=np.float32),                 # 1
            np.array([frame_h_score[0]], dtype=np.float32),              # 1
            np.array([frame_h_score[1]], dtype=np.float32),              # 1
            np.array([both_present], dtype=np.float32),                  # 1
            np.array([inter_dist], dtype=np.float32),                    # 1
            inter_offset                                                 # 2
        ], axis=0).astype(np.float32)
        
        telemetry = {
            "left_detected": left_detected,
            "right_detected": right_detected,
            "both_detected": (left_detected and right_detected),
            "left_score": float(frame_h_score[0]),
            "right_score": float(frame_h_score[1])
        }
        
        return flat_134, telemetry, results

    def reset_tracking(self):
        """Reset temporal Hungarian cache."""
        self.prev_wrists = None
        self.prev_presence = np.zeros(2, dtype=np.float32)

    def close(self):
        self.detector.close()


class LiveTemporalPreprocessor:
    """
    Standardizes a rolling 64-frame buffer and computes spike-free 129-D velocity features.
    """
    def __init__(self, scaler_path: str):
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler parameters not found at {scaler_path}")
        scaler = np.load(scaler_path)
        self.mean = scaler["mean"].astype(np.float32)
        self.std = scaler["std"].astype(np.float32)
        
    def process_window(self, window_134: np.ndarray) -> np.ndarray:
        """
        Transforms (64, 134) window into (64, 263) float32 tensor matching model input.
        """
        T, D = window_134.shape
        assert T == 64 and D == 134, f"Expected (64, 134), got ({T}, {D})"
        
        norm_seq = window_134.copy()
        p_L = norm_seq[:, 126] > 0.5
        p_R = norm_seq[:, 127] > 0.5
        p_both = norm_seq[:, 130] > 0.5
        
        # 1. Active-frame Z-Score Standardization (preserves 0-masks)
        if np.any(p_L):
            norm_seq[p_L, 0:63] = (norm_seq[p_L, 0:63] - self.mean[0:63]) / (self.std[0:63] + 1e-6)
        norm_seq[~p_L, 0:63] = 0.0
        
        if np.any(p_R):
            norm_seq[p_R, 63:126] = (norm_seq[p_R, 63:126] - self.mean[63:126]) / (self.std[63:126] + 1e-6)
        norm_seq[~p_R, 63:126] = 0.0
        
        if np.any(p_both):
            norm_seq[p_both, 131:134] = (norm_seq[p_both, 131:134] - self.mean[131:134]) / (self.std[131:134] + 1e-6)
        norm_seq[~p_both, 131:134] = 0.0
        
        # 2. First-Order Temporal Velocity with Transition Spike Prevention
        # Left Hand Velocity
        valid_L = np.zeros(T, dtype=bool)
        valid_L[1:] = p_L[1:] & p_L[:-1]
        v_L = np.zeros((T, 63), dtype=np.float32)
        v_L[1:] = np.where(valid_L[1:, None], norm_seq[1:, 0:63] - norm_seq[:-1, 0:63], 0.0)
        
        # Right Hand Velocity
        valid_R = np.zeros(T, dtype=bool)
        valid_R[1:] = p_R[1:] & p_R[:-1]
        v_R = np.zeros((T, 63), dtype=np.float32)
        v_R[1:] = np.where(valid_R[1:, None], norm_seq[1:, 63:126] - norm_seq[:-1, 63:126], 0.0)
        
        # Interaction Velocity
        valid_both = np.zeros(T, dtype=bool)
        valid_both[1:] = p_both[1:] & p_both[:-1]
        v_inter = np.zeros((T, 3), dtype=np.float32)
        v_inter[1:] = np.where(valid_both[1:, None], norm_seq[1:, 131:134] - norm_seq[:-1, 131:134], 0.0)
        
        # Concatenate: 134 + 63 + 63 + 3 = 263
        feat_263 = np.concatenate([norm_seq, v_L, v_R, v_inter], axis=-1).astype(np.float32)
        return feat_263


class PredictionStabilizer:
    """
    Temporal hysteresis and weighted voting filter for online predictions.
    Prevents single-frame classification flicker, eliminates stale predictions,
    and stabilizes the user-facing HUD.
    """
    def __init__(
        self,
        window_size: int = 5,
        confidence_threshold: float = 0.35,
        idle_dropout_frames: int = 12,
        uncertain_dropout_frames: int = 5
    ):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.idle_dropout_frames = idle_dropout_frames
        self.uncertain_dropout_frames = uncertain_dropout_frames
        
        self.history = deque(maxlen=window_size)
        self.consecutive_idle_frames = 0
        self.consecutive_uncertain_frames = 0
        
        self.current_stabilized_class: str = "IDLE"
        self.current_stabilized_id: int = -1
        self.current_stabilized_conf: float = 0.0

    def update(
        self,
        raw_pred: Optional[Dict[str, Any]],
        has_hands: bool
    ) -> Tuple[str, int, float]:
        """
        Ingest a new raw prediction and return (stabilized_class, stabilized_id, stabilized_conf).
        """
        if not has_hands:
            self.consecutive_idle_frames += 1
            if self.consecutive_idle_frames >= self.idle_dropout_frames:
                self.reset()
            return self.current_stabilized_class, self.current_stabilized_id, self.current_stabilized_conf
            
        self.consecutive_idle_frames = 0
        if raw_pred is None:
            return self.current_stabilized_class, self.current_stabilized_id, self.current_stabilized_conf
            
        p_class = raw_pred["predicted_class"]
        p_id = raw_pred["predicted_class_id"]
        p_conf = raw_pred["confidence"]
        
        # Only vote if confidence passes minimum floor
        if p_conf >= self.confidence_threshold:
            self.history.append((p_class, p_id, p_conf))
        else:
            self.history.append(("UNCERTAIN", -1, p_conf))
            
        if len(self.history) == 0:
            return self.current_stabilized_class, self.current_stabilized_id, self.current_stabilized_conf
            
        # Weighted confidence voting across window
        class_scores: Dict[Tuple[str, int], float] = {}
        for c_name, c_id, conf in self.history:
            if c_name == "UNCERTAIN":
                continue
            key = (c_name, c_id)
            class_scores[key] = class_scores.get(key, 0.0) + conf
            
        if not class_scores:
            # All frames in history are UNCERTAIN -> decay stale predictions once window is full
            if len(self.history) >= self.window_size:
                self.current_stabilized_class = "IDLE"
                self.current_stabilized_id = -1
                self.current_stabilized_conf = 0.0
            return self.current_stabilized_class, self.current_stabilized_id, self.current_stabilized_conf
            
        self.consecutive_uncertain_frames = 0
        
        # Find leading candidate
        best_candidate, best_score = max(class_scores.items(), key=lambda item: item[1])
        c_name, c_id = best_candidate
        
        # Count votes for leading candidate in history
        votes_count = sum(1 for name, _, _ in self.history if name == c_name)
        
        # Hysteresis rules:
        # Maintaining existing class requires standard threshold
        # Switching to a new class requires 2-frame consensus or strong mass (>= 2x threshold)
        if c_name == self.current_stabilized_class:
            can_switch = (best_score >= self.confidence_threshold)
        else:
            can_switch = (best_score >= (self.confidence_threshold * 1.5)) and (votes_count >= 2 or best_score >= (self.confidence_threshold * 2.0))
            
        if can_switch:
            self.current_stabilized_class = c_name
            self.current_stabilized_id = c_id
            # Average confidence of winning class within window
            matching_confs = [c for name, _, c in self.history if name == c_name]
            self.current_stabilized_conf = float(np.mean(matching_confs)) if matching_confs else float(p_conf)
            
        return self.current_stabilized_class, self.current_stabilized_id, self.current_stabilized_conf

    def reset(self):
        self.history.clear()
        self.consecutive_idle_frames = 0
        self.consecutive_uncertain_frames = 0
        self.current_stabilized_class = "IDLE"
        self.current_stabilized_id = -1
        self.current_stabilized_conf = 0.0


def draw_hud(
    frame: np.ndarray,
    tracking_status: str,
    raw_pred: Optional[Dict[str, Any]],
    stabilized_class: str,
    stabilized_conf: float,
    buffer_fill: int,
    fps: float,
    latencies: Dict[str, float],
    show_landmarks: bool = True,
    mp_results: Any = None,
    mp_drawing: Any = None,
    mp_hands: Any = None
) -> np.ndarray:
    """
    Render clean, engineering-oriented live telemetry HUD.
    """
    h, w, _ = frame.shape
    hud = frame.copy()
    
    # 1. Render Hand Landmarks if enabled
    if show_landmarks and mp_results and mp_results.multi_hand_landmarks and mp_drawing and mp_hands:
        for hand_landmarks in mp_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                hud,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )
            
    # 2. Top Header Bar (Semi-transparent dark overlay)
    overlay = hud.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 24, 28), -1)
    
    # Left Telemetry Panel
    cv2.rectangle(overlay, (12, 52), (320, 260), (20, 24, 28), -1)
    
    # Right Candidate Ranking Panel
    cv2.rectangle(overlay, (w - 300, 52), (w - 12, 210), (20, 24, 28), -1)
    
    # Bottom Latency Bar
    cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 24, 28), -1)
    
    # Blend overlay with 0.75 opacity
    cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)
    
    # Header Text
    cv2.putText(hud, "EchoGesture | INT8 Edge Pipeline", (15, 27), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 230, 255), 1, cv2.LINE_AA)
    fps_color = (0, 255, 0) if fps >= 25 else ((0, 200, 255) if fps >= 15 else (0, 0, 255))
    cv2.putText(hud, f"FPS: {fps:.1f}", (w - 110, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)
    
    # Left Telemetry Panel Contents
    status_color = (0, 255, 0) if "TRACKING" in tracking_status else (200, 200, 200)
    cv2.putText(hud, f"Status: {tracking_status}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)
    
    buf_color = (0, 255, 0) if buffer_fill == 64 else (0, 180, 255)
    cv2.putText(hud, f"Buffer: {buffer_fill}/64 frames", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, buf_color, 1, cv2.LINE_AA)
    
    # Primary Stabilized Prediction
    cv2.putText(hud, "STABILIZED SIGN:", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    stab_color = (0, 255, 100) if stabilized_class != "IDLE" else (120, 120, 120)
    cv2.putText(hud, stabilized_class.upper(), (25, 160), cv2.FONT_HERSHEY_DUPLEX, 0.95, stab_color, 2, cv2.LINE_AA)
    
    conf_text = f"Confidence: {stabilized_conf * 100.0:.1f}%" if stabilized_class != "IDLE" else "Confidence: --"
    cv2.putText(hud, conf_text, (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    
    # Raw Instantaneous Prediction
    raw_text = raw_pred["predicted_class"].upper() if raw_pred else "--"
    raw_conf = f"({raw_pred['confidence']*100:.1f}%)" if raw_pred else ""
    cv2.putText(hud, f"Raw Instant: {raw_text} {raw_conf}", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(hud, "Controls: [q] Exit | [r] Reset | [d] Toggle LMs", (25, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    
    # Right Candidate Ranking Panel
    cv2.putText(hud, "TOP CANDIDATES:", (w - 285, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    if raw_pred and "top_5" in raw_pred:
        for idx, cand in enumerate(raw_pred["top_5"][:3]):
            y_pos = 105 + idx * 32
            cand_name = cand["class_name"]
            cand_conf = cand["confidence"]
            cv2.putText(hud, f"{idx+1}. {cand_name}", (w - 285, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(hud, f"{cand_conf*100:.1f}%", (w - 75, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            # Progress bar
            bar_w = int(120 * cand_conf)
            cv2.rectangle(hud, (w - 200, y_pos - 8), (w - 200 + bar_w, y_pos - 2), (0, 200, 255), -1)
            
    # Bottom Latency Bar
    lat_text = (
        f"Latency: Capture={latencies.get('capture_ms', 0):.1f}ms | "
        f"MediaPipe={latencies.get('mediapipe_ms', 0):.1f}ms | "
        f"Preproc={latencies.get('preproc_ms', 0):.2f}ms | "
        f"INT8 TFLite={latencies.get('inference_ms', 0):.2f}ms | "
        f"Total={latencies.get('total_ms', 0):.1f}ms"
    )
    cv2.putText(hud, lat_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1, cv2.LINE_AA)
    
    return hud


def run_live_inference(
    camera_index: Union[int, str] = 0,
    width: int = 640,
    height: int = 480,
    fps_target: int = 30,
    smoothing_window: int = 5,
    confidence_threshold: float = 0.35,
    headless: bool = False,
    max_frames: Optional[int] = None,
    log_csv: Optional[str] = None,
    mirror_preview: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Main live inference execution loop.
    """
    repo_root = Path(__file__).resolve().parent.parent
    scaler_path = str(repo_root / "data" / "processed" / "sequences_v3" / "scaler_params.npz")
    model_path = str(repo_root / "models" / "quantized" / "echogesture_exp_d_int8.tflite")
    
    print("=== EchoGesture Phase 6A: Live Inference Pipeline ===")
    print(f"Camera Index:         {camera_index} ({width}x{height} @ {fps_target} FPS)")
    print(f"Scaler Parameters:    {scaler_path}")
    print(f"INT8 TFLite Model:    {model_path}")
    print(f"Smoothing Window:     {smoothing_window} frames (Threshold: {confidence_threshold*100:.0f}%)")
    print(f"Headless Mode:        {headless}")
    
    if mirror_preview is None:
        mirror_preview = isinstance(camera_index, int)
    print(f"Mirror Preview:       {mirror_preview} (Model input is always canonical un-flipped)")
    
    # 1. Initialize Components
    print("\nInitializing pipeline components...")
    extractor = LiveLandmarkExtractor()
    preprocessor = LiveTemporalPreprocessor(scaler_path)
    runtime = EchoGestureRuntime(model_path=model_path)
    stabilizer = PredictionStabilizer(
        window_size=smoothing_window,
        confidence_threshold=confidence_threshold
    )
    
    buffer_134 = deque(maxlen=64)
    
    # 2. Camera Setup
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open camera {camera_index}.")
        extractor.close()
        return {"status": "CAMERA_ERROR", "frames_processed": 0}
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    
    window_name = "EchoGesture - Live Sign Language Recognition"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
    print("[SUCCESS] Live inference initialized. Starting stream...\n")
    
    frame_counter = 0
    t_prev = time.perf_counter()
    fps_display = 0.0
    show_landmarks = True
    
    # Telemetry logging records
    latency_records = []
    
    try:
        while True:
            t_loop_start = time.perf_counter()
            
            # Stage A: Camera Capture
            t_cap_0 = time.perf_counter()
            ret, frame = cap.read()
            t_cap_ms = (time.perf_counter() - t_cap_0) * 1000.0
            
            if not ret or frame is None:
                print("[WARNING] Frame grab failed.")
                break
                
            frame_counter += 1
            
            # FPS Calculation
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt) if fps_display > 0 else (1.0 / dt)
                
            # Stage B: MediaPipe Landmark Extraction on canonical un-flipped frame
            t_mp_0 = time.perf_counter()
            flat_134, telemetry, mp_res = extractor.process_frame(frame)
            t_mp_ms = (time.perf_counter() - t_mp_0) * 1000.0
            
            # Append to rolling FIFO buffer
            buffer_134.append(flat_134)
            buf_len = len(buffer_134)
            
            has_hands = telemetry["left_detected"] or telemetry["right_detected"]
            
            # Tracking status text
            if telemetry["both_detected"]:
                status_str = "TRACKING (2 HANDS)"
            elif telemetry["left_detected"]:
                status_str = "TRACKING (LEFT HAND)"
            elif telemetry["right_detected"]:
                status_str = "TRACKING (RIGHT HAND)"
            elif buf_len < 64:
                status_str = f"BUFFERING ({buf_len}/64)"
            else:
                status_str = "IDLE (NO HANDS)"
                
            # Stage C & D: Preprocessing & INT8 Inference
            raw_pred = None
            t_preproc_ms = 0.0
            t_infer_ms = 0.0
            
            if buf_len >= 16:
                window_arr = np.array(buffer_134, dtype=np.float32)
                
                # Active frame count in window (presence flag check)
                active_frames = np.sum((window_arr[:, 126] > 0.5) | (window_arr[:, 127] > 0.5))
                
                if active_frames >= 12:
                    t_prep_0 = time.perf_counter()
                    # Resample incomplete windows (16 <= buf_len < 64) to 64 frames
                    if buf_len < 64:
                        resampled_134 = resample_sequence(window_arr, target_len=64)
                    else:
                        resampled_134 = window_arr
                        
                    # Stage C: Preprocessing & Velocity Extraction (263-D)
                    feat_263 = preprocessor.process_window(resampled_134)
                    t_preproc_ms = (time.perf_counter() - t_prep_0) * 1000.0
                    
                    # Stage D: Pure INT8 Inference via EchoGestureRuntime
                    raw_pred = runtime.predict(feat_263)
                    t_infer_ms = raw_pred["latency_ms"]
                else:
                    status_str = "IDLE (SPARSE BUFFER)"
                    
            # Stage E: Temporal Prediction Stabilization
            stab_class, stab_id, stab_conf = stabilizer.update(raw_pred, has_hands)
            
            t_total_ms = (time.perf_counter() - t_loop_start) * 1000.0
            
            latencies = {
                "capture_ms": t_cap_ms,
                "mediapipe_ms": t_mp_ms,
                "preproc_ms": t_preproc_ms,
                "inference_ms": t_infer_ms,
                "total_ms": t_total_ms
            }
            
            # Record telemetry sample for every frame
            latency_records.append({
                "frame": frame_counter,
                "fps": round(fps_display, 1),
                "capture_ms": round(t_cap_ms, 2),
                "mediapipe_ms": round(t_mp_ms, 2),
                "preproc_ms": round(t_preproc_ms, 3),
                "inference_ms": round(t_infer_ms, 3),
                "total_ms": round(t_total_ms, 2),
                "status": status_str,
                "stabilized_sign": stab_class,
                "confidence": round(stab_conf, 3)
            })
                
            # Stage F: Live HUD Rendering
            if not headless:
                base_frame = frame.copy()
                if show_landmarks and mp_res and mp_res.multi_hand_landmarks and extractor.mp_drawing and extractor.mp_hands:
                    for hand_landmarks in mp_res.multi_hand_landmarks:
                        extractor.mp_drawing.draw_landmarks(
                            base_frame,
                            hand_landmarks,
                            extractor.mp_hands.HAND_CONNECTIONS,
                            extractor.mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                            extractor.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                        )
                
                display_base = cv2.flip(base_frame, 1) if mirror_preview else base_frame
                
                hud_frame = draw_hud(
                    frame=display_base,
                    tracking_status=status_str,
                    raw_pred=raw_pred,
                    stabilized_class=stab_class,
                    stabilized_conf=stab_conf,
                    buffer_fill=buf_len,
                    fps=fps_display,
                    latencies=latencies,
                    show_landmarks=False
                )
                
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit signal received.")
                    break
                elif key == ord('r'):
                    buffer_134.clear()
                    stabilizer.reset()
                    extractor.reset_tracking()
                    print("Pipeline buffers reset by user.")
                elif key == ord('d'):
                    show_landmarks = not show_landmarks
                    print(f"Landmark display: {show_landmarks}")
                    
            if max_frames and frame_counter >= max_frames:
                print(f"Reached max_frames limit ({max_frames}).")
                break
                
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        extractor.close()
        print(f"Pipeline closed cleanly. Total frames processed: {frame_counter}")
        
    # Save latency report if requested
    if log_csv and latency_records:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        df_lat = pd.DataFrame(latency_records)
        df_lat.to_csv(log_csv, index=False)
        print(f"Saved latency measurements log to {log_csv}")
        
    return {
        "status": "SUCCESS",
        "frames_processed": frame_counter,
        "latency_records_count": len(latency_records)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Live Inference Pipeline")
    parser.add_argument("--camera", type=str, default="0", help="Camera index or video file path (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--smoothing_window", type=int, default=5, help="Stabilization window size (default: 5)")
    parser.add_argument("--confidence_threshold", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process before auto-exit")
    parser.add_argument("--log_csv", type=str, default="reports/phase6a_latency_report.csv", help="Latency CSV path")
    args = parser.parse_args()
    
    try:
        cam_source: Union[int, str] = int(args.camera)
    except ValueError:
        cam_source = args.camera

    run_live_inference(
        camera_index=cam_source,
        width=args.width,
        height=args.height,
        fps_target=args.fps,
        smoothing_window=args.smoothing_window,
        confidence_threshold=args.confidence_threshold,
        headless=args.headless,
        max_frames=args.max_frames,
        log_csv=args.log_csv
    )
