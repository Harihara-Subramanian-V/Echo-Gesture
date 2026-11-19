"""
EchoGesture Phase 6A: Live Camera Acquisition Module
Standalone OpenCV video capture utility for webcam verification and feed monitoring.

Features:
- Configurable camera device index, resolution, and FPS target
- Graceful failure handling if camera is unavailable or disconnected
- Real-time FPS overlay
- Clean shutdown on 'q', ESC, or interrupt signal
- Zero raw footage saving/uploading (strict privacy)
"""

import sys
import time
import argparse
from typing import Optional
import cv2


def run_live_camera(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    fps_target: int = 30
):
    """
    Open camera feed and display live video with diagnostic overlay.
    """
    print(f"=== EchoGesture Live Camera Feed ===")
    print(f"Attempting to open camera index {camera_index} at {width}x{height}...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open video device at camera index {camera_index}.")
        print("Possible causes:")
        print("  1. No webcam connected or device disabled.")
        print("  2. Camera is in use by another application.")
        print("  3. Camera index is different (try --camera 1 or --camera 2).")
        print("\nExiting gracefully.")
        return False
        
    # Configure capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_target)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[SUCCESS] Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    print("Press 'q' or 'ESC' in the camera window to exit.\n")
    
    window_name = "EchoGesture - Live Camera Feed (Press 'q' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    prev_time = time.perf_counter()
    fps_display = 0.0
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[WARNING] Failed to grab frame from camera stream. Exiting loop.")
                break
                
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            frame_count += 1
            
            # Exponential moving average for smooth FPS display
            if dt > 0:
                instant_fps = 1.0 / dt
                fps_display = 0.9 * fps_display + 0.1 * instant_fps if fps_display > 0 else instant_fps
                
            # Flip horizontally for natural mirror display
            display_frame = cv2.flip(frame, 1)
            
            # Simple diagnostic overlay
            overlay_text = f"Live Camera | {actual_width}x{actual_height} | FPS: {fps_display:.1f}"
            cv2.putText(
                display_frame,
                overlay_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                display_frame,
                "Press 'q' or ESC to exit",
                (15, actual_height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is ESC
                print("Exit signal received from user.")
                break
                
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera released and windows closed cleanly. Total frames processed: {frame_count}")
        
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoGesture Live Camera Utility")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    args = parser.parse_args()
    
    run_live_camera(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps_target=args.fps
    )
