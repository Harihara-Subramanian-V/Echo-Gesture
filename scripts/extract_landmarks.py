"""
EchoGesture: MediaPipe Landmark Extraction Pipeline (v3 - Final Validated)
Extracts 21 Left-Hand and 21 Right-Hand landmarks per frame from video files.
Saves structured feature tensors in compressed .npz format with comprehensive metadata.

Key Enhancements in v3:
1. Temporal Trajectory Tracking: Resolves duplicate/ambiguous handedness via Euclidean nearest-wrist
   matching against the preceding frame (t-1), preventing artificial slot jumping.
2. Correct Confidence Semantics: Documents and records handedness classification confidence (softmax
   probability p in [0, 1]) without confounding it with palm detection confidence.
3. Aspect-Ratio Corrected Geometry: Calculates isotropic, diagonal-normalized geometric inter-hand
   distance (d_geom) and 2D offsets (dx_geom, dy_geom), ensuring uniform scale across landscape,
   square, and portrait videos.
4. Strict Absence Masking: Missing hands strictly zero-padded with explicit presence = 0.0.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand landmarks from sign language videos.")
    parser.add_argument("--candidates_json", type=str, default="data/metadata/ms_asl/landmark_pilot_candidates.json",
                        help="Path to JSON file containing candidate videos to process.")
    parser.add_argument("--output_dir", type=str, default="data/processed/landmark_pilot",
                        help="Directory where extracted .npz files and summary will be saved.")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5,
                        help="MediaPipe minimum detection confidence threshold.")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="MediaPipe minimum tracking confidence threshold.")
    parser.add_argument("--model_complexity", type=int, default=1,
                        help="MediaPipe Hands model complexity (0=light, 1=full).")
    parser.add_argument("--static_image_mode", action="store_true", default=False,
                        help="Whether to treat each frame independently (slower, no temporal tracking). Default is False.")
    return parser.parse_args()


class LandmarkExtractor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 static_image_mode: bool = False):
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )

    def close(self):
        if self.detector:
            self.detector.close()

    def process_video(self, video_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Process all frames of a video and extract Left and Right hand landmarks.
        Returns a dict of numpy arrays and extraction statistics.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_diagonal = float(np.sqrt(width**2 + height**2)) if width > 0 and height > 0 else 1.0

        norm_landmarks_list = []
        world_landmarks_list = []
        presence_list = []
        handedness_scores_list = []
        both_present_list = []
        inter_hand_dist_geom_list = []
        inter_hand_offset_geom_list = []
        flat_features_list = []

        frames_with_left = 0
        frames_with_right = 0
        frames_with_both = 0
        frames_with_any = 0
        frames_with_none = 0
        left_h_scores = []
        right_h_scores = []

        # Previous frame wrist cache for temporal tracking: shape (2, 2) [slot 0: Left, slot 1: Right]
        prev_wrists = None
        prev_presence = np.zeros(2, dtype=np.float32)

        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.detector.process(rgb_frame)

            # Per-frame containers: Slot 0 = Left Hand, Slot 1 = Right Hand
            frame_norm = np.zeros((2, 21, 3), dtype=np.float32)
            frame_world = np.zeros((2, 21, 3), dtype=np.float32)
            frame_pres = np.zeros(2, dtype=np.float32)
            frame_h_score = np.zeros(2, dtype=np.float32)

            left_detected = False
            right_detected = False

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                # Collect detected hands with classification
                detected_hands = []
                for h_idx in range(num_hands):
                    h_lms = results.multi_hand_landmarks[h_idx]
                    h_wlms = results.multi_hand_world_landmarks[h_idx] if results.multi_hand_world_landmarks else None
                    handedness_info = results.multi_handedness[h_idx].classification[0]
                    label = handedness_info.label  # 'Left' or 'Right'
                    score = float(handedness_info.score)  # Handedness classification confidence

                    norm_coords = np.array([[lm.x, lm.y, lm.z] for lm in h_lms.landmark], dtype=np.float32)
                    world_coords = np.array([[lm.x, lm.y, lm.z] for lm in h_wlms.landmark], dtype=np.float32) if h_wlms else np.zeros((21, 3), dtype=np.float32)

                    detected_hands.append({
                        "label": label,
                        "score": score,
                        "norm": norm_coords,
                        "world": world_coords,
                        "wrist_xy": norm_coords[0, :2]
                    })

                # Resolve slot assignment (0: Viewer-Left Hand, 1: Viewer-Right Hand)
                if num_hands == 1:
                    h = detected_hands[0]
                    slot = 0 if h["label"] == "Left" else 1
                    frame_norm[slot] = h["norm"]
                    frame_world[slot] = h["world"]
                    frame_pres[slot] = 1.0
                    frame_h_score[slot] = h["score"]
                    if slot == 0:
                        left_detected = True
                        left_h_scores.append(h["score"])
                    else:
                        right_detected = True
                        right_h_scores.append(h["score"])

                elif num_hands >= 2:
                    left_candidates = [h for h in detected_hands if h["label"] == "Left"]
                    right_candidates = [h for h in detected_hands if h["label"] == "Right"]

                    if left_candidates and right_candidates:
                        # Clear distinct labels: select highest scoring candidates
                        best_left = max(left_candidates, key=lambda x: x["score"])
                        best_right = max(right_candidates, key=lambda x: x["score"])

                        frame_norm[0] = best_left["norm"]
                        frame_world[0] = best_left["world"]
                        frame_pres[0] = 1.0
                        frame_h_score[0] = best_left["score"]
                        left_detected = True
                        left_h_scores.append(best_left["score"])

                        frame_norm[1] = best_right["norm"]
                        frame_world[1] = best_right["world"]
                        frame_pres[1] = 1.0
                        frame_h_score[1] = best_right["score"]
                        right_detected = True
                        right_h_scores.append(best_right["score"])
                    else:
                        # Ambiguous duplicate labels: resolve via Temporal Trajectory Tracking if history exists
                        c0, c1 = detected_hands[0], detected_hands[1]
                        
                        if prev_wrists is not None and prev_presence[0] == 1.0 and prev_presence[1] == 1.0:
                            # Hungarian nearest-neighbor temporal trajectory matching against prev_wrists
                            # Cost direct: c0 -> Slot 0, c1 -> Slot 1
                            cost_direct = float(np.linalg.norm(c0["wrist_xy"] - prev_wrists[0]) + np.linalg.norm(c1["wrist_xy"] - prev_wrists[1]))
                            # Cost swap: c1 -> Slot 0, c0 -> Slot 1
                            cost_swap = float(np.linalg.norm(c1["wrist_xy"] - prev_wrists[0]) + np.linalg.norm(c0["wrist_xy"] - prev_wrists[1]))
                            
                            if cost_swap < cost_direct:
                                h_slot0, h_slot1 = c1, c0
                            else:
                                h_slot0, h_slot1 = c0, c1
                        else:
                            # Fallback if no 2-hand history: spatial horizontal order (smaller x = viewer left)
                            if c0["wrist_xy"][0] <= c1["wrist_xy"][0]:
                                h_slot0, h_slot1 = c0, c1
                            else:
                                h_slot0, h_slot1 = c1, c0

                        frame_norm[0] = h_slot0["norm"]
                        frame_world[0] = h_slot0["world"]
                        frame_pres[0] = 1.0
                        frame_h_score[0] = h_slot0["score"]
                        left_detected = True
                        left_h_scores.append(h_slot0["score"])

                        frame_norm[1] = h_slot1["norm"]
                        frame_world[1] = h_slot1["world"]
                        frame_pres[1] = 1.0
                        frame_h_score[1] = h_slot1["score"]
                        right_detected = True
                        right_h_scores.append(h_slot1["score"])

            # Update temporal wrist cache for subsequent frame tracking
            prev_wrists = np.array([frame_norm[0, 0, :2], frame_norm[1, 0, :2]], dtype=np.float32)
            prev_presence = frame_pres.copy()

            # Aspect-Ratio Corrected Geometric Inter-Hand Features (Isotropic in pixel space, normalized by frame diagonal)
            if left_detected and right_detected:
                # Pixel displacement: dx_px = (x_R - x_L) * width, dy_px = (y_R - y_L) * height
                dx_norm = float(frame_norm[1, 0, 0] - frame_norm[0, 0, 0])
                dy_norm = float(frame_norm[1, 0, 1] - frame_norm[0, 0, 1])
                dx_px = dx_norm * width
                dy_px = dy_norm * height
                
                # Diagonal-normalized geometric distance and directional displacement offsets
                dist_px = float(np.sqrt(dx_px**2 + dy_px**2))
                inter_hand_dist_geom = float(dist_px / frame_diagonal)
                inter_hand_offset_geom = np.array([dx_px / frame_diagonal, dy_px / frame_diagonal], dtype=np.float32)
                both_present = 1.0
            else:
                inter_hand_dist_geom = 0.0
                inter_hand_offset_geom = np.zeros(2, dtype=np.float32)
                both_present = 0.0

            # Statistics update
            if left_detected and right_detected:
                frames_with_both += 1
                frames_with_any += 1
            elif left_detected or right_detected:
                frames_with_any += 1
                if left_detected:
                    frames_with_left += 1
                else:
                    frames_with_right += 1
            else:
                frames_with_none += 1

            # Flat 134-D feature vector for sequence modeling:
            # - Left hand 21 normalized landmarks (x, y, z): 63 values (indices 0..62)
            # - Right hand 21 normalized landmarks (x, y, z): 63 values (indices 63..125)
            # - Left hand presence flag (1.0 or 0.0): 1 value (index 126)
            # - Right hand presence flag (1.0 or 0.0): 1 value (index 127)
            # - Left hand handedness classification confidence score ([0.0, 1.0]): 1 value (index 128)
            # - Right hand handedness classification confidence score ([0.0, 1.0]): 1 value (index 129)
            # - Both hands co-present interaction flag (1.0 or 0.0): 1 value (index 130)
            # - Aspect-ratio corrected diagonal-normalized Euclidean distance: 1 value (index 131)
            # - Aspect-ratio corrected diagonal-normalized horizontal offset dx_geom: 1 value (index 132)
            # - Aspect-ratio corrected diagonal-normalized vertical offset dy_geom: 1 value (index 133)
            # Total = 63 + 63 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 134 features
            flat_feat = np.concatenate([
                frame_norm[0].flatten(),                                     # 63
                frame_norm[1].flatten(),                                     # 63
                np.array([frame_pres[0]], dtype=np.float32),                 # 1
                np.array([frame_pres[1]], dtype=np.float32),                 # 1
                np.array([frame_h_score[0]], dtype=np.float32),              # 1
                np.array([frame_h_score[1]], dtype=np.float32),              # 1
                np.array([both_present], dtype=np.float32),                  # 1
                np.array([inter_hand_dist_geom], dtype=np.float32),          # 1
                inter_hand_offset_geom                                       # 2
            ], axis=0)
            assert flat_feat.shape[0] == 134, f"Expected 134 features, got {flat_feat.shape[0]}"

            norm_landmarks_list.append(frame_norm)
            world_landmarks_list.append(frame_world)
            presence_list.append(frame_pres)
            handedness_scores_list.append(frame_h_score)
            both_present_list.append(both_present)
            inter_hand_dist_geom_list.append(inter_hand_dist_geom)
            inter_hand_offset_geom_list.append(inter_hand_offset_geom)
            flat_features_list.append(flat_feat)

            frame_idx += 1

        cap.release()
        elapsed_sec = time.time() - start_time
        proc_fps = frame_idx / elapsed_sec if elapsed_sec > 0 else 0.0

        if frame_idx == 0:
            raise ValueError(f"No frames could be read from video: {video_path}")

        # Pack into full numpy arrays with documented semantics
        res_dict = {
            "norm_landmarks": np.stack(norm_landmarks_list, axis=0),              # (T, 2, 21, 3)
            "world_landmarks": np.stack(world_landmarks_list, axis=0),            # (T, 2, 21, 3)
            "presence": np.stack(presence_list, axis=0),                          # (T, 2)
            "handedness_score": np.stack(handedness_scores_list, axis=0),          # (T, 2)
            "confidence": np.stack(handedness_scores_list, axis=0),                # (T, 2) [Alias]
            "both_hands_present": np.array(both_present_list, dtype=np.float32),  # (T,)
            "inter_hand_distance_geom": np.array(inter_hand_dist_geom_list, dtype=np.float32), # (T,)
            "inter_hand_distance": np.array(inter_hand_dist_geom_list, dtype=np.float32),      # (T,) [Alias]
            "inter_hand_offset_geom": np.stack(inter_hand_offset_geom_list, axis=0),            # (T, 2)
            "inter_hand_offset_2d": np.stack(inter_hand_offset_geom_list, axis=0),              # (T, 2) [Alias]
            "flat_features": np.stack(flat_features_list, axis=0)                 # (T, 134)
        }

        stats = {
            "total_frames": frame_idx,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "duration_sec": round(frame_idx / fps, 2) if fps > 0 else 0.0,
            "frames_with_any_hand": frames_with_any,
            "frames_with_both_hands": frames_with_both,
            "frames_with_left_only": frames_with_left,
            "frames_with_right_only": frames_with_right,
            "frames_with_no_hand": frames_with_none,
            "any_hand_detection_rate": round(frames_with_any / frame_idx, 4),
            "both_hands_detection_rate": round(frames_with_both / frame_idx, 4),
            "left_hand_detection_rate": round((frames_with_both + frames_with_left) / frame_idx, 4),
            "right_hand_detection_rate": round((frames_with_both + frames_with_right) / frame_idx, 4),
            "mean_handedness_score_left": round(float(np.mean(left_h_scores)), 4) if left_h_scores else 0.0,
            "mean_handedness_score_right": round(float(np.mean(right_h_scores)), 4) if right_h_scores else 0.0,
            "processing_time_sec": round(elapsed_sec, 3),
            "processing_fps": round(proc_fps, 2),
            "extraction_status": "SUCCESS"
        }

        return res_dict, stats


def main():
    args = parse_args()
    print("=== EchoGesture: MediaPipe Landmark Extraction Pipeline (v3 - Final Validated) ===")
    print(f"Candidates file: {args.candidates_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence thresholds: detect={args.min_detection_confidence}, track={args.min_tracking_confidence}")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"Loaded {len(candidates)} candidate videos.")

    extractor = LandmarkExtractor(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        model_complexity=args.model_complexity,
        static_image_mode=args.static_image_mode
    )

    pilot_results = []
    total_start = time.time()

    for idx, cand in enumerate(candidates, 1):
        vid_id = cand["video_id"]
        gloss = cand["normalized_label"]
        vpath = cand["video_path"]
        split = cand["unified_split"]
        handedness = cand["handedness_type"]

        print(f"\n[{idx}/{len(candidates)}] Processing {vid_id} ('{gloss}', {handedness}, {split})...")
        npz_filename = f"{vid_id}.npz"
        npz_filepath = os.path.join(args.output_dir, npz_filename)

        try:
            t0 = time.time()
            data_dict, stats = extractor.process_video(vpath)

            # Save compressed npz with audited keys
            np.savez_compressed(
                npz_filepath,
                norm_landmarks=data_dict["norm_landmarks"],
                world_landmarks=data_dict["world_landmarks"],
                presence=data_dict["presence"],
                handedness_score=data_dict["handedness_score"],
                confidence=data_dict["confidence"],
                both_hands_present=data_dict["both_hands_present"],
                inter_hand_distance_geom=data_dict["inter_hand_distance_geom"],
                inter_hand_distance=data_dict["inter_hand_distance"],
                inter_hand_offset_geom=data_dict["inter_hand_offset_geom"],
                inter_hand_offset_2d=data_dict["inter_hand_offset_2d"],
                flat_features=data_dict["flat_features"]
            )

            file_size_bytes = os.path.getsize(npz_filepath)
            print(f"  -> Extracted {stats['total_frames']} frames in {stats['processing_time_sec']}s ({stats['processing_fps']} FPS)")
            print(f"  -> Any hand: {stats['any_hand_detection_rate']*100:.1f}%, Both: {stats['both_hands_detection_rate']*100:.1f}%, Left: {stats['left_hand_detection_rate']*100:.1f}%, Right: {stats['right_hand_detection_rate']*100:.1f}%")
            print(f"  -> Saved {npz_filepath} ({file_size_bytes/1024:.1f} KB)")

            result_entry = {
                **cand,
                "npz_file": npz_filename,
                "npz_path": npz_filepath,
                "npz_size_bytes": file_size_bytes,
                **stats
            }
            pilot_results.append(result_entry)

        except Exception as e:
            print(f"  ERROR processing {vpath}: {str(e)}")
            failed_entry = {
                **cand,
                "npz_file": None,
                "npz_path": None,
                "npz_size_bytes": 0,
                "extraction_status": "FAILED",
                "error_message": str(e)
            }
            pilot_results.append(failed_entry)

    extractor.close()
    total_elapsed = time.time() - total_start

    # Save summary manifest
    manifest_out = os.path.join(args.output_dir, "pilot_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(pilot_results, f, indent=2)

    # Save summary CSV
    csv_out = os.path.join(args.output_dir, "pilot_summary.csv")
    fieldnames = [
        "video_id", "dataset", "normalized_label", "original_label", "signer_id",
        "unified_split", "handedness_type", "resolution", "fps", "duration_sec",
        "total_frames", "frames_with_any_hand", "frames_with_both_hands",
        "frames_with_left_only", "frames_with_right_only", "frames_with_no_hand",
        "any_hand_detection_rate", "both_hands_detection_rate",
        "left_hand_detection_rate", "right_hand_detection_rate",
        "mean_handedness_score_left", "mean_handedness_score_right",
        "processing_time_sec", "processing_fps", "npz_size_bytes", "extraction_status"
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in pilot_results:
            writer.writerow(r)

    print(f"\n=== Final Validated Pilot Extraction Completed in {total_elapsed:.2f}s ===")
    print(f"Manifest written to: {manifest_out}")
    print(f"CSV summary written to: {csv_out}")


if __name__ == "__main__":
    main()
