# EchoGesture: Demo Recording Operational Checklist

**Project:** EchoGesture (Track 2: Samsung XR Hackathon 2026)  
**Deliverable:** Pre-Flight, Live Recording & Post-Production Quality Checklist  
**Target Device:** `vivo V2416` (`V2418i`) running Android 16 (API 36)  
**Max Allowable Video Length:** **3 minutes 00 seconds** (Target ceiling: **2 minutes 40 seconds**)  
**Video Resolution Target:** 1080p FHD (1920x1080 or 1080x1920 / 1080x2400) at 30/60 FPS  

---

## 1. Physical Hardware & Device Pre-Flight

Complete every item in this section prior to starting the video recording:

- [ ] **Physical Battery Level:** Ensure device battery is charged to $\ge 70\%$ (minimum requirement: $> 50\%$) to prevent OS battery saver throttling or low-power clock down-clocking.
- [ ] **Developer Settings & UI Hygiene:**
  - [ ] Enable *Developer Options* $\to$ *Demo Mode* (or set clock to standard time e.g. 10:00, battery icon 100%, clean notification bar).
  - [ ] Disable all background chat apps, pop-up notifications, and floating chat bubbles.
  - [ ] Ensure "Show taps" is toggled **ON** in Developer Options so touch interactions (`btnSwitchCamera`, `btnToggleSpeech`, `btnReplay`) are visually clear to viewers.
  - [ ] Set Screen Timeout to $\ge 10\text{ minutes}$ (or "Stay awake while charging").
- [ ] **Android Text-To-Speech (TTS) Verification:**
  - [ ] Ensure Google TTS engine is set to default in Android Settings (*System* $\to$ *Accessibility* $\to$ *Text-to-speech output*).
  - [ ] Verify volume slider for *Media* and *Accessibility / Assistance* is set to $85\%\text{--}100\%$.
  - [ ] Verify offline speech pack for `English (United States)` is downloaded locally.
- [ ] **Camera Lens & Mounting:**
  - [ ] Clean both rear 50 MP and front 32 MP camera lenses with a microfiber cloth to prevent glare or smudge distortion.
  - [ ] Secure device in a stable phone clamp or tabletop tripod with clear line-of-sight to the signer's torso and hands.
- [ ] **Quick Settings Configuration:**
  - [ ] Ensure the **Airplane Mode** quick-toggle tile is positioned on the first swipe page of Quick Settings for rapid demonstration during Scene 6.

---

## 2. Audio & Visual Studio Setup

- [ ] **Lighting & Background:**
  - [ ] Dual-point diffuse lighting facing the presenter/signer (prevent harsh shadows on fingers).
  - [ ] Avoid heavy backlighting (windows or bright overhead lamps behind the signer) to prevent palm detector underexposure.
  - [ ] Contrast-appropriate background: Plain solid background (neutral gray, off-white, or dark blue) to ensure maximum contrast for hand silhouettes.
- [ ] **Screen Recording & External Capture:**
  - [ ] **Primary Capture (Screen Recorder):** Use built-in Android screen recorder or ADB screenrecord:
    ```powershell
    adb shell screenrecord --bit-rate 16000000 --time-limit 180 /sdcard/echogesture_demo.mp4
    ```
    *Optionally capture internal system audio directly.*
  - [ ] **Secondary Capture (Over-the-Shoulder / Third-Person):** A secondary 4K or 1080p camera framing the signer, desk, and physical phone in real time to prove zero latency between physical hands and on-screen HUD.
- [ ] **Voice Narration Audio:**
  - [ ] Dedicated cardioid condenser or lavalier microphone for voiceover narration.
  - [ ] Record voiceover in a sound-dampened room (noise floor $< -50\text{ dBFS}$).

---

## 3. Gestural Sequence & Vocabulary Rehearsal

Rehearse each of the 6 core demonstration signs to guarantee $>95\%$ instant stabilizer lock:

| Sign ID | Target Gesture | Type | Kinematic Execution Cue | Expected Latency | Audio Cue |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **1** | **`DRINK`** | Single-Hand | Right hand forms loose "C" shape, tilts toward mouth in natural drinking gesture. Hold for 1.0 second. | $< 100\text{ ms}$ | *"Drink"* |
| **2** | **`COMPUTER`** | Single-Hand | Right hand forms "C" shape, moves horizontally along left wrist/forearm. | $< 100\text{ ms}$ | *"Computer"* |
| **3** | **`BEFORE`** | Single-Hand | Right open palm faces backward over right shoulder, gesturing backward. | $< 100\text{ ms}$ | *"Before"* |
| **4** | **`BOOK`** | Two-Hand | Both open palms together at midline, thumbs upward, then open book-like. | $< 100\text{ ms}$ | *"Book"* |
| **5** | **`CLOTHES`** | Two-Hand | Open 5-hands brush downward along upper chest/collarbone twice. | $< 100\text{ ms}$ | *"Clothes"* |
| **6** | **`FINISH`** | Two-Hand | Open 5-hands with palms facing signer flick outward so palms face outward. | $< 100\text{ ms}$ | *"Finish"* |

- [ ] **Depth Invariance Test:** During the **`FINISH`** sign, rehearse starting at $0.3\text{ m}$ ($>50\%$ frame height) and smoothly retreating to $1.2\text{ m}$ ($<15\%$ frame height) to demonstrate robust palm-scale normalization.
- [ ] **Cooldown Cadence:** Maintain at least a 1.5-second pause between distinct signs to respect the 1500 ms speech cooldown gate without triggering premature suppression.

---

## 4. Live Recording Run Sheet & Timing Checklist

Adhere strictly to the scene durations. The hard limit for Track 2 submission is 3:00 minutes. Target 2:40 for safety:

```
[ ] 0:00 - 0:20 (20s): Scene 1 — Problem Statement & Edge Architecture
    Check: Lower third appears; clear statement of "100% offline, zero cloud".
[ ] 0:20 - 0:45 (25s): Scene 2 — Cold Launch & 60 FPS HUD
    Check: Tap app icon; preview opens at 60 FPS; show IDLE state with infMs=0.00.
[ ] 0:45 - 1:25 (40s): Scene 3 — Single-Hand Signs ("DRINK", "COMPUTER", "BEFORE")
    Check: Green skeleton lock; confidence >95%; native TTS announces each sign.
[ ] 1:25 - 2:05 (40s): Scene 4 — Two-Hand Signs ("BOOK", "CLOTHES", "FINISH")
    Check: Dual cyan/magenta skeletons; Hungarian slot preservation; near/far depth demo.
[ ] 2:05 - 2:25 (20s): Scene 5 — Ergonomic Features (Lens Flip, Mute, Replay)
    Check: Tap camera flip (front selfie mirrored); tap MUTE (silent); tap REPLAY (audio).
[ ] 2:25 - 2:40 (15s): Scene 6 — Airplane Mode Verification & Metrics
    Check: Quick Settings swipe; Airplane mode ON; sign "YES"; closing metrics slide.
```

- [ ] **Total Elapsed Time Check:** Stop recording at or before **02:40**. Confirm total duration does not exceed **02:50** under any circumstances.

---

## 5. Post-Production & Quality Sign-Off

- [ ] **Audio Balance & Sync:**
  - [ ] Verify narration vocals are clean and mixed at $-14\text{ LUFS}$ to $-16\text{ LUFS}$.
  - [ ] Verify phone TTS speech is crisp, perfectly synchronized with on-screen HUD text, and not drowned out by voiceover or background music.
- [ ] **Visual Clarity:**
  - [ ] HUD metrics (`FPS: 60`, `Model: 0.6ms`, `Confidence: >95%`) are sharp, legible, and uncompressed.
  - [ ] Hand landmark skeleton overlays align with physical hand positions without frame stutter.
- [ ] **Export Specifications:**
  - [ ] Format: MP4 container (`H.264` / `AAC`).
  - [ ] Resolution: $1920 \times 1080$ (16:9) or $1080 \times 1920$ (9:16 vertical).
  - [ ] File size: $< 250\text{ MB}$ for seamless submission upload.
- [ ] **Storage Location:** Save final video render to:
  `c:\EchoCam\submission\demo\echogesture_demo_final.mp4`
