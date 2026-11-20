package com.echogesture.app.feature

import kotlin.math.sqrt

/**
 * Represents a single detected hand candidate from an upstream detector (e.g., MediaPipe Hands).
 *
 * @property label Handedness label predicted by detector ("Left" or "Right").
 * @property score Handedness classification confidence in [0.0, 1.0].
 * @property landmarks63 21 landmarks flattened as [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] (size 63).
 * @property wristX Wrist X coordinate (defaults to landmarks63[0]).
 * @property wristY Wrist Y coordinate (defaults to landmarks63[1]).
 */
data class DetectedHandCandidate(
    val label: String,           // "Left" or "Right"
    val score: Float,            // Handedness confidence in [0.0, 1.0]
    val landmarks63: FloatArray, // 21 (x, y, z) normalized landmarks (size 63)
    val wristX: Float = landmarks63[0],
    val wristY: Float = landmarks63[1]
) {
    init {
        require(landmarks63.size == HandSlotResolver.HAND_FEATURE_DIM) {
            "landmarks63 must have size ${HandSlotResolver.HAND_FEATURE_DIM}, got ${landmarks63.size}"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DetectedHandCandidate

        if (label != other.label) return false
        if (score != other.score) return false
        if (!landmarks63.contentEquals(other.landmarks63)) return false
        if (wristX != other.wristX) return false
        if (wristY != other.wristY) return false

        return true
    }

    override fun hashCode(): Int {
        var result = label.hashCode()
        result = 31 * result + score.hashCode()
        result = 31 * result + landmarks63.contentHashCode()
        result = 31 * result + wristX.hashCode()
        result = 31 * result + wristY.hashCode()
        return result
    }

    override fun toString(): String {
        return "DetectedHandCandidate(label='$label', score=$score, wristX=$wristX, wristY=$wristY)"
    }
}

/**
 * Resolved canonical hand slots for downstream feature vector assembly and model inference.
 *
 * Slot 0 is strictly designated for the Viewer-Left Hand (subject's right hand in selfie mode).
 * Slot 1 is strictly designated for the Viewer-Right Hand (subject's left hand in selfie mode).
 *
 * @property slot0Landmarks Viewer-Left Hand 21 landmarks (size 63, all zeros if absent).
 * @property slot1Landmarks Viewer-Right Hand 21 landmarks (size 63, all zeros if absent).
 * @property presLeft 1.0f if Viewer-Left Hand detected, 0.0f otherwise.
 * @property presRight 1.0f if Viewer-Right Hand detected, 0.0f otherwise.
 * @property scoreLeft Confidence score for Viewer-Left Hand (0.0f if absent).
 * @property scoreRight Confidence score for Viewer-Right Hand (0.0f if absent).
 */
data class ResolvedHandSlots(
    val slot0Landmarks: FloatArray, // Viewer-Left Hand (size 63)
    val slot1Landmarks: FloatArray, // Viewer-Right Hand (size 63)
    val presLeft: Float,            // 1.0f or 0.0f
    val presRight: Float,           // 1.0f or 0.0f
    val scoreLeft: Float,           // Score or 0.0f
    val scoreRight: Float           // Score or 0.0f
) {
    init {
        require(slot0Landmarks.size == HandSlotResolver.HAND_FEATURE_DIM) {
            "slot0Landmarks must have size ${HandSlotResolver.HAND_FEATURE_DIM}, got ${slot0Landmarks.size}"
        }
        require(slot1Landmarks.size == HandSlotResolver.HAND_FEATURE_DIM) {
            "slot1Landmarks must have size ${HandSlotResolver.HAND_FEATURE_DIM}, got ${slot1Landmarks.size}"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as ResolvedHandSlots

        if (!slot0Landmarks.contentEquals(other.slot0Landmarks)) return false
        if (!slot1Landmarks.contentEquals(other.slot1Landmarks)) return false
        if (presLeft != other.presLeft) return false
        if (presRight != other.presRight) return false
        if (scoreLeft != other.scoreLeft) return false
        if (scoreRight != other.scoreRight) return false

        return true
    }

    override fun hashCode(): Int {
        var result = slot0Landmarks.contentHashCode()
        result = 31 * result + slot1Landmarks.contentHashCode()
        result = 31 * result + presLeft.hashCode()
        result = 31 * result + presRight.hashCode()
        result = 31 * result + scoreLeft.hashCode()
        result = 31 * result + scoreRight.hashCode()
        return result
    }

    override fun toString(): String {
        return "ResolvedHandSlots(presLeft=$presLeft, presRight=$presRight, scoreLeft=$scoreLeft, scoreRight=$scoreRight)"
    }
}

/**
 * Pure Kotlin hand slot resolver implementing the exact canonical assignment logic
 * from `LiveLandmarkExtractor` in `scripts/live_inference.py`.
 *
 * Responsibilities:
 * 1. Direct single-hand assignment (Left -> Slot 0, Right -> Slot 1).
 * 2. Multi-hand distinct label assignment (best Left -> Slot 0, best Right -> Slot 1 by confidence score).
 * 3. Ambiguous duplicate label resolution:
 *    - Temporal Hungarian tracking when both hands were present in the immediate previous frame.
 *    - Spatial horizontal fallback (smaller wrist X -> Slot 0, larger wrist X -> Slot 1).
 * 4. Empty candidate zero-masking.
 * 5. Temporal tracking cache maintenance and reset.
 */
class HandSlotResolver {

    /**
     * Internal temporal tracking cache for wrist (x, y) coordinates.
     * Index 0: Slot 0 (Viewer-Left Hand) wrist [x, y]
     * Index 1: Slot 1 (Viewer-Right Hand) wrist [x, y]
     */
    var prevWrists: Array<FloatArray>? = null
        internal set

    /**
     * Internal temporal tracking cache for hand presence flags in previous frame.
     * Index 0: Slot 0 presence (1.0f or 0.0f)
     * Index 1: Slot 1 presence (1.0f or 0.0f)
     */
    val prevPresence: FloatArray = FloatArray(2)

    /**
     * Resolves a list of detected hand candidates into canonical slots.
     */
    fun resolve(candidates: List<DetectedHandCandidate>): ResolvedHandSlots {
        val slot0Landmarks = FloatArray(HAND_FEATURE_DIM)
        val slot1Landmarks = FloatArray(HAND_FEATURE_DIM)
        var presLeft = 0.0f
        var presRight = 0.0f
        var scoreLeft = 0.0f
        var scoreRight = 0.0f

        when {
            candidates.isEmpty() -> {
                // Both slots strictly all zeros
            }
            candidates.size == 1 -> {
                val candidate = candidates[0]
                if (candidate.label.equals(LABEL_LEFT, ignoreCase = true)) {
                    System.arraycopy(candidate.landmarks63, 0, slot0Landmarks, 0, HAND_FEATURE_DIM)
                    presLeft = 1.0f
                    scoreLeft = candidate.score
                } else if (candidate.label.equals(LABEL_RIGHT, ignoreCase = true)) {
                    System.arraycopy(candidate.landmarks63, 0, slot1Landmarks, 0, HAND_FEATURE_DIM)
                    presRight = 1.0f
                    scoreRight = candidate.score
                } else {
                    // Fallback for unlabeled/unknown candidates by horizontal wrist coordinate
                    if (candidate.wristX <= 0.5f) {
                        System.arraycopy(candidate.landmarks63, 0, slot0Landmarks, 0, HAND_FEATURE_DIM)
                        presLeft = 1.0f
                        scoreLeft = candidate.score
                    } else {
                        System.arraycopy(candidate.landmarks63, 0, slot1Landmarks, 0, HAND_FEATURE_DIM)
                        presRight = 1.0f
                        scoreRight = candidate.score
                    }
                }
            }
            else -> {
                val leftCands = candidates.filter { it.label.equals(LABEL_LEFT, ignoreCase = true) }
                val rightCands = candidates.filter { it.label.equals(LABEL_RIGHT, ignoreCase = true) }

                if (leftCands.isNotEmpty() && rightCands.isNotEmpty()) {
                    val bestLeft = leftCands.maxByOrNull { it.score } ?: leftCands[0]
                    val bestRight = rightCands.maxByOrNull { it.score } ?: rightCands[0]

                    System.arraycopy(bestLeft.landmarks63, 0, slot0Landmarks, 0, HAND_FEATURE_DIM)
                    presLeft = 1.0f
                    scoreLeft = bestLeft.score

                    System.arraycopy(bestRight.landmarks63, 0, slot1Landmarks, 0, HAND_FEATURE_DIM)
                    presRight = 1.0f
                    scoreRight = bestRight.score
                } else {
                    // Ambiguous duplicate labels (both Left or both Right)
                    val c0 = candidates[0]
                    val c1 = candidates[1]

                    val (hSlot0, hSlot1) = if (prevWrists != null && prevPresence[0] == 1.0f && prevPresence[1] == 1.0f) {
                        val prev0 = prevWrists!![0]
                        val prev1 = prevWrists!![1]

                        val costDirect = euclideanDistance(c0.wristX, c0.wristY, prev0[0], prev0[1]) +
                                euclideanDistance(c1.wristX, c1.wristY, prev1[0], prev1[1])
                        val costSwap = euclideanDistance(c1.wristX, c1.wristY, prev0[0], prev0[1]) +
                                euclideanDistance(c0.wristX, c0.wristY, prev1[0], prev1[1])

                        if (costSwap < costDirect) {
                            Pair(c1, c0)
                        } else {
                            Pair(c0, c1)
                        }
                    } else {
                        // Horizontal fallback: smaller wrist X -> Slot 0, larger wrist X -> Slot 1
                        if (c0.wristX <= c1.wristX) {
                            Pair(c0, c1)
                        } else {
                            Pair(c1, c0)
                        }
                    }

                    System.arraycopy(hSlot0.landmarks63, 0, slot0Landmarks, 0, HAND_FEATURE_DIM)
                    presLeft = 1.0f
                    scoreLeft = hSlot0.score

                    System.arraycopy(hSlot1.landmarks63, 0, slot1Landmarks, 0, HAND_FEATURE_DIM)
                    presRight = 1.0f
                    scoreRight = hSlot1.score
                }
            }
        }

        // Update tracking cache
        prevWrists = arrayOf(
            floatArrayOf(slot0Landmarks[0], slot0Landmarks[1]),
            floatArrayOf(slot1Landmarks[0], slot1Landmarks[1])
        )
        prevPresence[0] = presLeft
        prevPresence[1] = presRight

        return ResolvedHandSlots(
            slot0Landmarks = slot0Landmarks,
            slot1Landmarks = slot1Landmarks,
            presLeft = presLeft,
            presRight = presRight,
            scoreLeft = scoreLeft,
            scoreRight = scoreRight
        )
    }

    /**
     * Overload accepting vararg candidates for convenience.
     */
    fun resolve(vararg candidates: DetectedHandCandidate): ResolvedHandSlots =
        resolve(candidates.toList())

    /**
     * Resets the temporal tracking cache.
     */
    fun reset() {
        prevWrists = null
        prevPresence[0] = 0.0f
        prevPresence[1] = 0.0f
    }

    companion object {
        const val LANDMARKS_PER_HAND = 21
        const val COORDS_PER_LANDMARK = 3
        const val HAND_FEATURE_DIM = 63 // 21 * 3
        const val LABEL_LEFT = "Left"
        const val LABEL_RIGHT = "Right"

        private fun euclideanDistance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
            val dx = x1 - x2
            val dy = y1 - y2
            return sqrt(dx * dx + dy * dy)
        }
    }
}
