# EchoGesture — Phase 2: Vocabulary Selection Report
**Project:** EchoGesture (Samsung XR Hackathon 2026 — Track 2)  
**Date:** 2026-11-19  
**Status:** Vocabulary Selection Completed  
**Lead Model:** Gemini 3.8 Flash (High)  

---

## 1. Executive Summary

This report establishes the curated **40-class initial sign language vocabulary** for the EchoGesture recognition engine. 

Rather than choosing signs based solely on conversational popularity or subjective preference, this lexicon was derived systematically from the verified **WLASL benchmark metadata** (`WLASL_v0.3.json`) according to four strict engineering criteria:
1. **High Sample Density:** Every class contains at least **20 verified video instances** (up to 40 instances; total = 911 instances).
2. **High Signer Diversity:** Every class features between **11 and 21 distinct signers** (far exceeding the project minimum threshold of $\ge 3$ signers).
3. **Balanced Bilateral vs. Unilateral Morphology:** 21 two-handed signs (52.5%) and 19 one-handed signs (47.5%) spanning diverse spatial articulation zones.
4. **Minimal Confusion Risk:** Avoiding phonetically ambiguous minimal pairs (signs differing only in subtle finger contact) to ensure high initial discriminability for lightweight sequence models.

---

## 2. Master Table: Selected 40-Sign Lexicon

| Class Index | Gloss Label | Total Instances | Distinct Signers | Morphology | Primary Articulation Zone | Kinematic Trajectory & Visual Signature |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **0** | `book` | 40 | 15 | Two-handed | Chest / Neutral | Palms together opening outward like opening book pages. |
| **1** | `drink` | 35 | 18 | One-handed | Mouth / Chin | C-hand tilting upward toward mouth in drinking motion. |
| **2** | `computer` | 30 | 15 | Two-handed | Chest | Dominant C-hand tracing circular arc along forearm. |
| **3** | `before` | 26 | 16 | Two-handed | Torso | Flat open hand retracting backward toward chest. |
| **4** | `chair` | 26 | 16 | Two-handed | Neutral Space | Repetitive double-tapping of bent U-fingers onto base hand. |
| **5** | `go` | 26 | 17 | Two-handed | Neutral Space | Index fingers flipping forward in outward arc trajectory. |
| **6** | `clothes` | 25 | 12 | Two-handed | Upper Chest | Open 5-hands brushing downward along chest repeatedly. |
| **7** | `who` | 25 | 21 | One-handed | Chin | Thumb on chin while index finger curls repeatedly in L-shape. |
| **8** | `candy` | 24 | 16 | One-handed | Cheek | Index fingertip twisting into cheek in localized rotation. |
| **9** | `cousin` | 23 | 13 | One-handed | Temple / Ear | C-hand shaking with rotational oscillation near temple. |
| **10** | `deaf` | 23 | 18 | One-handed | Cheek / Ear | Index finger touching ear then moving forward to touch mouth. |
| **11** | `fine` | 22 | 16 | One-handed | Chest | 5-hand thumb tapping center of chest with fingers splayed. |
| **12** | `help` | 22 | 15 | Two-handed | Neutral Space | Flat non-dominant palm lifting an upward-thrusting dominant fist. |
| **13** | `no` | 22 | 18 | One-handed | Neutral Space | Index and middle fingers snapping firmly against thumb. |
| **14** | `thin` | 22 | 17 | Two-handed | Neutral Space | Both G-hands pulling vertically apart with dynamic velocity. |
| **15** | `walk` | 22 | 18 | Two-handed | Lower Space | Flat palms alternating front-to-back like stepping feet. |
| **16** | `year` | 22 | 16 | Two-handed | Neutral Space | Dominant fist orbiting around and landing on stationary fist. |
| **17** | `yes` | 22 | 17 | One-handed | Neutral Space | S-fist nodding vertically like an affirmative head nod. |
| **18** | `all` | 21 | 15 | Two-handed | Neutral Space | Dominant hand sweeping in full circle into non-dominant palm. |
| **19** | `black` | 21 | 17 | One-handed | Forehead | Index finger drawing horizontally across forehead. |
| **20** | `cool` | 21 | 11 | Two-handed | Chest | Bent 5-hands fanning inward toward chest repeatedly. |
| **21** | `finish` | 21 | 16 | Two-handed | Neutral Space | 5-hands snapping outward with sudden open-palm release. |
| **22** | `hot` | 21 | 17 | One-handed | Mouth | Claw hand pulling abruptly away from mouth downward. |
| **23** | `like` | 21 | 14 | One-handed | Chest | 8-hand pulling outward from chest pinching thumb and middle. |
| **24** | `many` | 21 | 14 | Two-handed | Neutral Space | S-fists bursting upward into wide open 5-hands. |
| **25** | `mother` | 21 | 15 | One-handed | Chin | Open 5-hand thumb tapping chin repeatedly. |
| **26** | `now` | 21 | 15 | Two-handed | Neutral Space | Y-hands dropping downward in synchrony with firm stop. |
| **27** | `orange` | 21 | 17 | One-handed | Chin | C-hand repeatedly squeezing into S-fist directly at chin. |
| **28** | `table` | 21 | 14 | Two-handed | Neutral Space | Horizontal forearms tapping down on base forearm. |
| **29** | `thanksgiving` | 21 | 14 | One-handed | Chin / Chest | Flat hand extending forward from chin and chest twice. |
| **30** | `what` | 21 | 18 | Two-handed | Neutral Space | Open palms facing upward oscillating side-to-side horizontally. |
| **31** | `woman` | 21 | 15 | One-handed | Chin / Chest | 5-hand thumb touching chin then trailing down to sternum. |
| **32** | `bed` | 20 | 16 | Two-handed | Head / Cheek | Both palms flat together resting against tilted cheek. |
| **33** | `blue` | 20 | 17 | One-handed | Neutral Space | B-hand oscillating with horizontal wrist rotation. |
| **34** | `bowling` | 20 | 13 | One-handed | Lower Space | Curved hand executing an underhand swinging arc. |
| **35** | `can` | 20 | 13 | Two-handed | Neutral Space | S-fists thrusting downward simultaneously into a locked pose. |
| **36** | `dog` | 20 | 17 | One-handed | Hip / Leg | Hand patting thigh then snapping fingers outward. |
| **37** | `family` | 20 | 16 | Two-handed | Neutral Space | F-hands tracing forward circle and joining at pinky edges. |
| **38** | `fish` | 20 | 17 | One-handed | Neutral Space | Flat B-hand undulating horizontally forward like a fish. |
| **39** | `graduate` | 20 | 14 | Two-handed | Neutral Space | G-hand making circular flourish and landing on base palm. |

---

## 3. Class Imbalance & Distribution Analysis

### Instance Distribution Statistics
- **Total Selected Vocabulary:** 40 Classes
- **Total Video Samples:** Exactly **911 video clips**
- **Maximum Samples per Class:** 40 instances (`book`)
- **Minimum Samples per Class:** 20 instances (e.g., `bed`, `blue`, `can`, `fish`, `graduate`)
- **Mean Samples per Class:** $22.78 \pm 3.92$
- **Median Samples per Class:** 21 instances

### Imbalance Ratio Calculation
$$\text{Imbalance Ratio} = \frac{\max(N_c)}{\min(N_c)} = \frac{40}{20} = 2.00$$

> [!TIP]
> **Imbalance Health:**  
> In natural video datasets, imbalance ratios commonly exceed $10.0$ or $50.0$. An imbalance ratio of **$2.00$** is exceptionally tight and ensures that cross-entropy loss gradients will not be monopolized by dominant classes. Standard class-weighted cross-entropy will comfortably normalize the gradient contributions during training.

---

## 4. Morphological and Spatial Breakdown

To ensure that the model learns robust spatial and temporal motion features rather than relying on a single visual pattern, the 40 classes are deliberately balanced across morphological categories:

### A. Morphological Distribution
- **Two-Handed Signs (21 classes, 52.5%):**  
  `book`, `computer`, `before`, `chair`, `go`, `clothes`, `help`, `thin`, `walk`, `year`, `all`, `cool`, `finish`, `many`, `now`, `table`, `what`, `bed`, `can`, `family`, `graduate`.  
  *Advantage:* Stresses bilateral coordinate tracking and relative hand-to-hand spatial encoding.
- **One-Handed Signs (19 classes, 47.5%):**  
  `drink`, `who`, `candy`, `cousin`, `deaf`, `fine`, `no`, `yes`, `black`, `hot`, `like`, `mother`, `orange`, `thanksgiving`, `woman`, `blue`, `bowling`, `dog`, `fish`.  
  *Advantage:* Stresses single-hand trajectory encoding, wrist rotation, and body-relative position against upper torso/facial landmarks.

### B. Articulation Space Distribution
- **Head & Facial Contact (11 classes):** `drink`, `who`, `candy`, `cousin`, `deaf`, `black`, `hot`, `mother`, `orange`, `woman`, `bed`.
- **Chest & Upper Torso (10 classes):** `book`, `computer`, `before`, `clothes`, `fine`, `cool`, `like`, `thanksgiving`, `can`.
- **Neutral / Peripheral Space (19 classes):** `chair`, `go`, `help`, `no`, `thin`, `walk`, `year`, `yes`, `all`, `finish`, `many`, `now`, `table`, `what`, `blue`, `bowling`, `dog`, `family`, `fish`, `graduate`.

---

## 5. Visual Discriminability & Risk Management

1. **Minimal Pairs Filtered Out:**  
   We investigated potentially confusable pairs (e.g., *apple* vs. *candy*, which share similar cheek articulation). While *candy* is retained (Rank 9, 24 instances), *apple* was omitted from the initial 40 to prevent early baseline confusion caused by subtle knuckle differences before the feature normalization pipeline is tuned.
2. **Robustness to Hand Occlusion:**  
   Classes involving hand-over-hand interaction (`help`, `table`, `year`, `chair`) were specifically audited. The presence of upper-body pose landmarks (shoulders, elbows, wrists) provides continuous spatial context even when one hand temporarily occludes the other.

---

*This vocabulary selection report completes Task 2 of Phase 2. The vocabulary mapping has been synchronized with the project configuration.*
