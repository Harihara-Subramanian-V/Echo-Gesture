import sys
import numpy as np
import cv2
import mediapipe as mp
import yt_dlp

def test_environment():
    print("=== ENVIRONMENT VERIFICATION ===")
    print(f"Python: {sys.version}")
    print(f"MediaPipe version: {mp.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"yt-dlp version: {yt_dlp.version.__version__}")
    
    print("\n--- 1. Testing MediaPipe Hands Initialization ---")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("MediaPipe Hands initialized successfully.")
    
    print("\n--- 2. Testing Synthetic Frame Inference ---")
    # Create a synthetic 480x640x3 RGB test image
    synthetic_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple shape
    cv2.circle(synthetic_img, (320, 240), 50, (255, 255, 255), -1)
    results_synthetic = hands.process(synthetic_img)
    print(f"Synthetic image processed. multi_hand_landmarks: {results_synthetic.multi_hand_landmarks}")
    
    print("\n--- 3. Testing Real Video Frame Inference ---")
    # Read frames from an actual downloaded video
    test_video_path = "data/raw/ms_asl/msasl_train_0000_J7tP98oDxqE.mp4"
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {test_video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {test_video_path} opened successfully. Total frames: {total_frames}")
    
    detected_hands_count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_video = hands.process(rgb_frame)
        if results_video.multi_hand_landmarks:
            detected_hands_count += 1
            if detected_hands_count == 1:
                print(f"First detection at frame {frame_idx}: {len(results_video.multi_hand_landmarks)} hand(s)")
                for idx, hand_lms in enumerate(results_video.multi_hand_landmarks):
                    handedness = results_video.multi_handedness[idx].classification[0].label
                    score = results_video.multi_handedness[idx].classification[0].score
                    print(f"  Hand {idx+1}: {handedness} (score={score:.4f}, landmarks={len(hand_lms.landmark)})")
                    wrist = hand_lms.landmark[0]
                    print(f"    Wrist landmark: x={wrist.x:.4f}, y={wrist.y:.4f}, z={wrist.z:.4f}")
        frame_idx += 1
    cap.release()
    print(f"Total processed frames: {frame_idx}, Frames with detected hands: {detected_hands_count}/{frame_idx} ({detected_hands_count/max(frame_idx,1)*100:.1f}%)")
    hands.close()
    
    print("\n--- 4. Checking yt-dlp functionality ---")
    ydl = yt_dlp.YoutubeDL({'quiet': True})
    print("yt-dlp initialized successfully.")
    
    print("\n=== ALL CHECKS PASSED ===")
    return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
