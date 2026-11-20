# EchoGesture: Video Demonstration Recording Script

**Project:** EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Deliverable:** 3-Minute Video Demo Script & Narration Blueprint  
**Target Duration:** **2 minutes 40 seconds** (Strictly within the 3:00 minute ceiling)  
**Execution Environment:** Physical Android Target (`vivo V2416` / Android 16 API 36)  
**Hardware Audio:** Internal Device Audio + Clean Vocal Narration Track  

---

## 1. Video Overview & Master Timing Breakdown

| Timecode | Segment Name | Segment Goal | On-Screen Visual | Spoken Audio / Narration |
| :---: | :--- | :--- | :--- | :--- |
| **0:00 – 0:20**<br>*(20 sec)* | **Architecture & Edge Pitch** | Introduce EchoGesture & offline edge paradigm | Title card $\to$ Physical phone running EchoGesture in hand | Pitch: 100% on-device edge AI, zero cloud dependencies, complete privacy, $<1\text{ ms}$ neural execution. |
| **0:20 – 0:45**<br>*(25 sec)* | **Cold Launch & 60 FPS HUD** | Demonstrate launch, CameraX 60 FPS VSYNC, HUD overlay | Tap icon $\to$ camera opens instantly $\to$ HUD displays 60 FPS, 0 hands `IDLE` state | Cold launch speed, smooth CameraX preview, non-blocking asynchronous analysis pipeline. |
| **0:45 – 1:25**<br>*(40 sec)* | **Single-Hand Sign Recognition** | Demonstrate real-time recognition & offline TTS voice | Present "DRINK", "COMPUTER", "BEFORE" gestures to camera | Live skeleton tracking, stabilized prediction, immediate voice synthesis from phone speaker. |
| **1:25 – 2:05**<br>*(40 sec)* | **Two-Hand Signs & Hungarian Slot Tracking** | Demonstrate dual-hand signs and distance invariance | Present "BOOK", "CLOTHES", "FINISH" gestures at various distances | Dual 21-joint skeleton tracking, Hungarian matching, depth-invariant palm normalization. |
| **2:05 – 2:25**<br>*(20 sec)* | **Ergonomic Features & Controls** | Showcase lens flip, speech mute/unmute, replay | Flip camera front/back, tap `SPEECH: MUTE`, tap `REPLAY` | Seamless camera switching with preview mirroring parity, full speech coordinator controls. |
| **2:25 – 2:40**<br>*(15 sec)* | **Offline Proof & Closing Metrics** | Prove 100% offline resilience and highlight metrics | Swipe down QS $\to$ Airplane Mode ON $\to$ sign "THANK YOU" / "YES" | Zero cloud calls, 135 KB model footprint, 0.599 ms inference. Clean sign-off. |

---

## 2. Detailed Scene-by-Scene Script

### Scene 1: The Edge Imperative & Architecture (0:00 – 0:20 | 20s)

**Visual:**
- Camera starts on a clean, well-lit desk.
- A physical Android phone (`vivo V2416`) sits in a stand, showing the home screen with the **EchoGesture** icon.
- Lower-third graphic appears: *"EchoGesture: 100% Offline Real-Time Sign Recognition for Edge & XR"*.

**Spoken Narration (Voiceover):**
> "Sign language is dynamic, expressive, and immediate. Yet most assistive tools rely on latency-heavy cloud APIs that compromise privacy and fail without internet.  
> Welcome to EchoGesture—a production-grade, 100% offline edge recognition engine built for the next generation of mobile and XR hardware. Everything you are about to see runs entirely on this physical smartphone, with zero server calls."

---

### Scene 2: Camera Launch & 60 FPS Live HUD (0:20 – 0:45 | 25s)

**Visual:**
- Presenter taps the EchoGesture icon on the phone.
- The app cold launches instantaneously. Camera preview springs to life without delay.
- The camera points at a neutral background.
- On-Screen HUD highlights:
  - Top badge: `FPS: 60.0 | VSYNC Locked`
  - Guidance badge: `WAITING FOR HANDS`
  - Hero Sign: `IDLE` (Confidence: `0.0%`)
  - Latency: `Model: 0.6ms | Pipeline: Idle`
- Presenter waves empty hand briefly out of frame to prove neural inference is bypassed when zero hands are visible.

**Spoken Narration (Voiceover):**
> "Launching EchoGesture: the CameraX engine locks immediately to 60 frames per second display VSYNC.  
> Notice our dual-pipeline decoupling: the vision analysis runs asynchronously on a dedicated worker thread. When no hands are in frame, neural inference is dynamically bypassed, drawing zero wasted CPU cycles."

---

### Scene 3: Single-Hand Signs & Offline Voice Playback (0:45 – 1:25 | 40s)

**Visual:**
- Presenter brings right hand into view.
- 21-joint skeleton immediately anchors over the hand with green joints and connecting bones.
- **Sign 1 ("DRINK"):**
  - Presenter forms the ASL sign for "DRINK" (curled fingers tilting toward mouth).
  - The HUD transitions: `COLLECTING_FRAMES` $\to$ `ACTIVE_PREDICTION`.
  - Hero text updates boldly to: **`DRINK`** (Confidence: `99.2%`).
  - Top candidates update: `1. DRINK (99%) | 2. WATER (1%)`.
  - *Audio:* The phone's native Google TTS voice announces clearly: *"Drink"*.
- **Sign 2 ("COMPUTER"):**
  - Presenter forms the ASL sign for "COMPUTER" (C-handshape moving along forearm).
  - Hero text snaps to: **`COMPUTER`** (Confidence: `99.6%`).
  - *Audio:* The phone announces: *"Computer"*.
- **Sign 3 ("BEFORE"):**
  - Presenter executes "BEFORE" (open palm gesturing backwards over shoulder).
  - Hero text snaps to: **`BEFORE`** (Confidence: `96.8%`).
  - *Audio:* The phone announces: *"Before"*.

**Spoken Narration (Voiceover):**
> "Presenting single-hand signs: watch the sub-millisecond skeleton lock.  
> As I sign *'Drink'*, the 64-frame rolling temporal buffer extracts 263 normalized features. The native hysteresis stabilizer locks the prediction, and Android's offline text-to-speech speaks the word instantly.  
> Next, *'Computer'*—detected and synthesized with over 99% confidence.  
> And *'Before'*. Notice the strict duplicate suppression: holding a sign produces exactly one clear voice announcement."

---

### Scene 4: Dual-Hand Signs, Hungarian Slotting & Distance Invariance (1:25 – 2:05 | 40s)

**Visual:**
- Presenter brings both hands into the frame.
- The HUD overlays two distinct skeletons:
  - Hand 0 (Viewer-Left): Cyan joints.
  - Hand 1 (Viewer-Right): Magenta joints.
- **Sign 4 ("BOOK"):**
  - Presenter executes the ASL sign for "BOOK" (palms together opening outward like a book).
  - Hero text confirms: **`BOOK`** (Confidence: `98.4%`).
  - *Audio:* Phone announces: *"Book"*.
- **Sign 5 ("CLOTHES"):**
  - Presenter demonstrates "CLOTHES" (open 5-hands brushing downward on chest).
  - Hero text confirms: **`CLOTHES`** (Confidence: `95.1%`).
  - *Audio:* Phone announces: *"Clothes"*.
- **Sign 6 ("FINISH"):**
  - Presenter demonstrates "FINISH" (5-hands flicking outward).
  - Presenter moves hands close to camera (filling 60% of frame), then moves back to arm's length (filling 15% of frame).
  - Hero text confirms: **`FINISH`** (Confidence: `97.8%`).
  - *Audio:* Phone announces: *"Finish"*.

**Spoken Narration (Voiceover):**
> "Now, two-handed signs. Our Hungarian slot resolver tracks viewer-left and viewer-right hands independently, populating eight inter-wrist spatial features.  
> Here is *'Book'*—dual skeletons tracked in real time.  
> Here is *'Clothes'*. Even with overlapping arm motions, our slot tracking prevents identity swaps.  
> And here is *'Finish'*. Watch as I move from close-up to arm's length: our palm-scale normalization makes gesture recognition completely invariant to camera distance."

---

### Scene 5: Ergonomic Controls: Camera Flip, Mute & Replay (2:05 – 2:25 | 20s)

**Visual:**
- Presenter demonstrates controls on the bottom control bar:
  1. **Camera Flip:** Tap `btnSwitchCamera`. The rear camera switches to the front selfie camera. The camera preview automatically mirrors for user comfort, while the underlying mathematical frames remain unmirrored for ML parity. Skeletons track seamlessly in selfie mode. Tap again to return to rear.
  2. **Speech Mute:** Tap `btnToggleSpeech`. The button updates to `SPEECH: OFF` and the status badge shows `Speech: Muted`. Presenter performs a sign; HUD updates visually, but audio stays silent.
  3. **Replay:** Tap `btnToggleSpeech` to unmute, then tap `btnReplay`. The phone re-speaks the last confirmed word on demand.

**Spoken Narration (Voiceover):**
> "EchoGesture is engineered for real-world ergonomics.  
> Switching between rear and front cameras is seamless—with automatic selfie mirroring on the display while preserving canonical mathematical orientation in the ML pipeline.  
> In quiet environments, one tap mutes voice feedback. And with on-demand replay, users can repeat the last recognized sign at any time."

---

### Scene 6: Offline Verification & Efficiency Closing (2:25 – 2:40 | 15s)

**Visual:**
- Presenter swipes down Android Quick Settings from the top of the screen.
- Presenter explicitly toggles **Airplane Mode ON** (Wi-Fi off, Mobile Data off, Bluetooth off).
- Swipes back up to EchoGesture.
- Presenter signs "YES" or "FAMILY".
- The HUD immediately detects **`YES`** (Confidence: `99.8%`) and the phone announces *"Yes"*.
- The screen displays final performance metric summary graphics:
  - **Neural Model Latency:** `0.599 ms` (LiteRT INT8)
  - **Model File Size:** `135.4 KB`
  - **Memory PSS:** `190 MB (Zero Leaks)`
  - **Offline Status:** `100% Autonomous Edge AI`

**Spoken Narration (Voiceover):**
> "Finally, the true edge test: with Airplane Mode active and zero network connectivity, EchoGesture recognizes and speaks with zero latency penalty.  
> A 135-kilobyte INT8 neural model running in just 0.6 milliseconds.  
> EchoGesture: accessible, private, and instantaneous communication on the edge."

---

## 3. Teleprompter & Presenter Cues

```text
[0:00] STAND IN FRONT OF CAMERA - PHONE IN STAND VISIBLE
[0:05] CUE INTRO PITCH: "100% offline edge recognition..."
[0:20] TAP APP ICON - POINT OUT 60 FPS HUD AND IDLE ZERO-HAND STATE
[0:45] SIGN: "DRINK" (Tilt cup to lips) -> WAIT FOR AUDIO "DRINK"
[0:58] SIGN: "COMPUTER" (C-hand forward along arm) -> WAIT FOR AUDIO "COMPUTER"
[1:12] SIGN: "BEFORE" (Palm sweeps backward over shoulder) -> WAIT FOR AUDIO "BEFORE"
[1:25] BRING BOTH HANDS IN -> SIGN "BOOK" (Open palms) -> WAIT FOR AUDIO "BOOK"
[1:38] SIGN: "CLOTHES" (5-hands sweep chest) -> WAIT FOR AUDIO "CLOTHES"
[1:50] SIGN: "FINISH" (Flick hands outward) -> MOVE NEAR/FAR TO SHOW DEPTH INVARIANCE
[2:05] TAP CAMERA FLIP -> SHOW SELFIE TRACKING -> TAP REVERSE BACK TO REAR
[2:15] TAP MUTE -> SIGN BRIEFLY (SILENT) -> UNMUTE -> TAP REPLAY ("FINISH")
[2:25] SWIPE QUICK SETTINGS -> AIRPLANE MODE ON -> SIGN "YES" -> INSTANT SPEECH
[2:35] DISPLAY FINAL METRICS SLIDE -> FADE OUT AT 2:40
```
