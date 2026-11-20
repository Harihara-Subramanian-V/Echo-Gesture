package com.echogesture.app.feature

import kotlin.math.sqrt

/**
 * 134-D Feature Extractor for EchoGesture Phase 7B.2.
 *
 * Implements the exact 134-dimensional feature vector assembly matching the canonical Python
 * implementation in `scripts/live_inference.py` (LiveLandmarkExtractor.process_frame) and
 * the Stage 1 Android Parity contract.
 *
 * Vector Layout (134-D):
 * - [0:63]   : Slot 0 (Viewer-Left Hand) 21 normalized landmarks (x, y, z)
 * - [63:126] : Slot 1 (Viewer-Right Hand) 21 normalized landmarks (x, y, z)
 * - [126]    : Left Hand Presence flag (1.0f or 0.0f)
 * - [127]    : Right Hand Presence flag (1.0f or 0.0f)
 * - [128]    : Left Hand confidence score (in [0.0, 1.0])
 * - [129]    : Right Hand confidence score (in [0.0, 1.0])
 * - [130]    : Both Hands Co-presence flag (1.0f if presLeft > 0.5f && presRight > 0.5f, else 0.0f)
 * - [131]    : Aspect-ratio corrected diagonal-normalized inter-wrist Euclidean distance:
 *              d_geom = sqrt((dx_norm * width)^2 + (dy_norm * height)^2) / sqrt(width^2 + height^2)
 * - [132:134]: Aspect-ratio corrected diagonal-normalized inter-wrist offsets (dx_px / diag, dy_px / diag)
 *
 * Zero-masking rules:
 * - If both hands are NOT present (presLeft <= 0.5f or presRight <= 0.5f):
 *   indices 130, 131, 132, 133 are strictly 0.0f.
 * - Inactive hand slot landmarks are strictly 0.0f.
 * - Inactive hand presence and score are strictly 0.0f.
 * - When neither hand is present, all 134 values are strictly 0.0f.
 */
class FeatureExtractor134D(
    private val preallocateOutput: Boolean = false
) {
    companion object {
        const val FEATURE_DIM = 134
        const val LANDMARKS_PER_HAND = 21
        const val COORDS_PER_HAND = 63 // 21 * 3

        const val SLOT0_START = 0
        const val SLOT0_END = 63
        const val SLOT1_START = 63
        const val SLOT1_END = 126

        const val INDEX_PRES_LEFT = 126
        const val INDEX_PRES_RIGHT = 127
        const val INDEX_SCORE_LEFT = 128
        const val INDEX_SCORE_RIGHT = 129
        const val INDEX_BOTH_PRESENT = 130
        const val INDEX_INTER_DIST = 131
        const val INDEX_INTER_OFFSET_DX = 132
        const val INDEX_INTER_OFFSET_DY = 133

        private const val PRESENCE_THRESHOLD = 0.5f
    }

    private val reusableVector = FloatArray(FEATURE_DIM)

    /**
     * Extracts canonical 134-D feature vector from resolved hand slots.
     *
     * @param slots Resolved hand slots containing slot0 (left) and slot1 (right) landmarks.
     * @param imageWidth Frame width in pixels.
     * @param imageHeight Frame height in pixels.
     * @return 134-D float array containing normalized landmarks, presence flags, scores, and inter-wrist geometry.
     */
    fun extract(slots: ResolvedHandSlots, imageWidth: Int, imageHeight: Int): FloatArray {
        val out = if (preallocateOutput) reusableVector else FloatArray(FEATURE_DIM)
        extractTo(slots, imageWidth, imageHeight, out)
        return out
    }

    /**
     * Backward-compatible alias for [extract] matching pipeline spec naming.
     */
    fun assemble(slots: ResolvedHandSlots, imageWidth: Int, imageHeight: Int): FloatArray {
        return extract(slots, imageWidth, imageHeight)
    }

    /**
     * Extracts canonical 134-D feature vector directly into the provided destination array
     * to avoid memory allocations in hot execution loops.
     *
     * @param slots Resolved hand slots.
     * @param imageWidth Frame width in pixels.
     * @param imageHeight Frame height in pixels.
     * @param destination Destination array of size at least 134.
     * @return The destination array.
     */
    fun extractTo(
        slots: ResolvedHandSlots,
        imageWidth: Int,
        imageHeight: Int,
        destination: FloatArray
    ): FloatArray {
        require(destination.size >= FEATURE_DIM) {
            "Destination array size must be at least $FEATURE_DIM, got ${destination.size}"
        }
        require(slots.slot0Landmarks.size == COORDS_PER_HAND) {
            "slot0Landmarks size must be $COORDS_PER_HAND, got ${slots.slot0Landmarks.size}"
        }
        require(slots.slot1Landmarks.size == COORDS_PER_HAND) {
            "slot1Landmarks size must be $COORDS_PER_HAND, got ${slots.slot1Landmarks.size}"
        }

        val leftPresent = slots.presLeft > PRESENCE_THRESHOLD
        val rightPresent = slots.presRight > PRESENCE_THRESHOLD

        // Slot 0: Viewer-Left Hand (indices 0..62)
        if (leftPresent) {
            System.arraycopy(slots.slot0Landmarks, 0, destination, SLOT0_START, COORDS_PER_HAND)
            destination[INDEX_PRES_LEFT] = 1.0f
            destination[INDEX_SCORE_LEFT] = slots.scoreLeft
        } else {
            destination.fill(0.0f, SLOT0_START, SLOT0_END)
            destination[INDEX_PRES_LEFT] = 0.0f
            destination[INDEX_SCORE_LEFT] = 0.0f
        }

        // Slot 1: Viewer-Right Hand (indices 63..125)
        if (rightPresent) {
            System.arraycopy(slots.slot1Landmarks, 0, destination, SLOT1_START, COORDS_PER_HAND)
            destination[INDEX_PRES_RIGHT] = 1.0f
            destination[INDEX_SCORE_RIGHT] = slots.scoreRight
        } else {
            destination.fill(0.0f, SLOT1_START, SLOT1_END)
            destination[INDEX_PRES_RIGHT] = 0.0f
            destination[INDEX_SCORE_RIGHT] = 0.0f
        }

        // Both hands co-presence and geometric inter-wrist features (indices 130..133)
        if (leftPresent && rightPresent && imageWidth > 0 && imageHeight > 0) {
            val w = imageWidth.toDouble()
            val h = imageHeight.toDouble()
            val diag = sqrt(w * w + h * h)

            if (diag > 0.0) {
                // Inter-wrist offsets: Slot 1 (Viewer-Right) - Slot 0 (Viewer-Left)
                val dxNorm = (slots.slot1Landmarks[0] - slots.slot0Landmarks[0]).toDouble()
                val dyNorm = (slots.slot1Landmarks[1] - slots.slot0Landmarks[1]).toDouble()

                val dxPx = dxNorm * w
                val dyPx = dyNorm * h
                val distPx = sqrt(dxPx * dxPx + dyPx * dyPx)

                destination[INDEX_BOTH_PRESENT] = 1.0f
                destination[INDEX_INTER_DIST] = (distPx / diag).toFloat()
                destination[INDEX_INTER_OFFSET_DX] = (dxPx / diag).toFloat()
                destination[INDEX_INTER_OFFSET_DY] = (dyPx / diag).toFloat()
            } else {
                destination[INDEX_BOTH_PRESENT] = 1.0f
                destination[INDEX_INTER_DIST] = 0.0f
                destination[INDEX_INTER_OFFSET_DX] = 0.0f
                destination[INDEX_INTER_OFFSET_DY] = 0.0f
            }
        } else {
            destination[INDEX_BOTH_PRESENT] = 0.0f
            destination[INDEX_INTER_DIST] = 0.0f
            destination[INDEX_INTER_OFFSET_DX] = 0.0f
            destination[INDEX_INTER_OFFSET_DY] = 0.0f
        }

        return destination
    }
}
