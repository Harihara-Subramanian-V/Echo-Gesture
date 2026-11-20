package com.echogesture.app.preprocessing

/**
 * Extracts 129-D first-order temporal velocity features from a standardized 134-D sequence.
 *
 * Implements transition-spike prevention matching Python `LiveTemporalPreprocessor.process_window`
 * and `scripts/augment_sequences.py:compute_velocity_features`:
 * - Boundary condition: Frame 0 velocity is strictly 0.0f across all 129 channels.
 * - Transition spike prevention: For frames t in 1 until T, velocity between frames t and t-1
 *   is computed if and only if the corresponding hand/interaction is present in BOTH frames.
 *   If either frame lacks presence (> 0.5f), velocity is strictly clamped to 0.0f.
 *
 * Channel Layout (129-D):
 * - [0..62]: Left hand landmark velocities (21 x 3 = 63 channels)
 * - [63..125]: Right hand landmark velocities (21 x 3 = 63 channels)
 * - [126..128]: Two-hand interaction velocities (wrist distance, dx, dy = 3 channels)
 */
class VelocityFeatureExtractor {

    /**
     * Extracts 129-D velocity features from a standardized sequence of shape [T][134].
     *
     * @param normSeq Standardized sequence of shape [T][134] (typically T=64).
     * @param destination Optional preallocated destination buffer of shape [T][129] to eliminate GC allocations.
     * @return Velocity features of shape [T][129].
     */
    fun extract(
        normSeq: Array<FloatArray>,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> {
        val tCount = normSeq.size
        if (tCount == 0) {
            return destination ?: Array(0) { FloatArray(VELOCITY_DIM) }
        }

        val vel = destination ?: Array(tCount) { FloatArray(VELOCITY_DIM) }
        require(vel.size == tCount) {
            "Destination size (${vel.size}) does not match sequence size ($tCount)"
        }

        // Frame 0 boundary condition: strictly 0.0 across all 129 channels
        require(normSeq[0].size == INPUT_FEATURE_DIM) {
            "Frame 0 feature dimension ${normSeq[0].size} does not match expected $INPUT_FEATURE_DIM"
        }
        require(vel[0].size == VELOCITY_DIM) {
            "Destination frame 0 dimension ${vel[0].size} does not match expected $VELOCITY_DIM"
        }
        vel[0].fill(0.0f)

        // Frames 1 until T
        for (t in 1 until tCount) {
            val cur = normSeq[t]
            val prev = normSeq[t - 1]
            val velT = vel[t]

            require(cur.size == INPUT_FEATURE_DIM) {
                "Frame $t feature dimension ${cur.size} does not match expected $INPUT_FEATURE_DIM"
            }
            require(velT.size == VELOCITY_DIM) {
                "Destination frame $t dimension ${velT.size} does not match expected $VELOCITY_DIM"
            }

            // 1. Left Hand Velocity (channels 0..62)
            val validL = (cur[INDEX_LEFT_PRESENT] > PRESENCE_THRESHOLD) &&
                         (prev[INDEX_LEFT_PRESENT] > PRESENCE_THRESHOLD)
            if (validL) {
                for (c in 0 until 63) {
                    velT[c] = cur[c] - prev[c]
                }
            } else {
                for (c in 0 until 63) {
                    velT[c] = 0.0f
                }
            }

            // 2. Right Hand Velocity (channels 63..125)
            val validR = (cur[INDEX_RIGHT_PRESENT] > PRESENCE_THRESHOLD) &&
                         (prev[INDEX_RIGHT_PRESENT] > PRESENCE_THRESHOLD)
            if (validR) {
                for (c in 0 until 63) {
                    velT[63 + c] = cur[63 + c] - prev[63 + c]
                }
            } else {
                for (c in 0 until 63) {
                    velT[63 + c] = 0.0f
                }
            }

            // 3. Interaction Velocity (channels 126..128)
            val validBoth = (cur[INDEX_BOTH_PRESENT] > PRESENCE_THRESHOLD) &&
                            (prev[INDEX_BOTH_PRESENT] > PRESENCE_THRESHOLD)
            if (validBoth) {
                for (c in 0 until 3) {
                    velT[126 + c] = cur[131 + c] - prev[131 + c]
                }
            } else {
                for (c in 0 until 3) {
                    velT[126 + c] = 0.0f
                }
            }
        }

        return vel
    }

    /**
     * Concatenates standardized sequence [normSeq] (134-D) and velocity sequence [velSeq] (129-D)
     * into a single [263-D] tensor matching the LiteRT model input format.
     *
     * @param normSeq Standardized sequence of shape [T][134].
     * @param velSeq Velocity feature sequence of shape [T][129].
     * @param destination Optional preallocated destination buffer of shape [T][263].
     * @return Full combined tensor of shape [T][263].
     */
    fun concatenate(
        normSeq: Array<FloatArray>,
        velSeq: Array<FloatArray>,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> {
        val tCount = normSeq.size
        require(velSeq.size == tCount) {
            "Sequence length mismatch: normSeq has $tCount frames, velSeq has ${velSeq.size}"
        }
        val out = destination ?: Array(tCount) { FloatArray(TOTAL_FEATURE_DIM) }
        require(out.size == tCount) {
            "Destination size (${out.size}) does not match sequence size ($tCount)"
        }

        for (t in 0 until tCount) {
            require(normSeq[t].size == INPUT_FEATURE_DIM) {
                "normSeq frame $t size ${normSeq[t].size} != $INPUT_FEATURE_DIM"
            }
            require(velSeq[t].size == VELOCITY_DIM) {
                "velSeq frame $t size ${velSeq[t].size} != $VELOCITY_DIM"
            }
            require(out[t].size == TOTAL_FEATURE_DIM) {
                "out frame $t size ${out[t].size} != $TOTAL_FEATURE_DIM"
            }
            System.arraycopy(normSeq[t], 0, out[t], 0, INPUT_FEATURE_DIM)
            System.arraycopy(velSeq[t], 0, out[t], INPUT_FEATURE_DIM, VELOCITY_DIM)
        }
        return out
    }

    companion object {
        const val INPUT_FEATURE_DIM = 134
        const val VELOCITY_DIM = 129
        const val TOTAL_FEATURE_DIM = 263
        const val SEQUENCE_LENGTH = 64
        const val PRESENCE_THRESHOLD = 0.5f

        const val INDEX_LEFT_PRESENT = 126
        const val INDEX_RIGHT_PRESENT = 127
        const val INDEX_BOTH_PRESENT = 130
    }
}
