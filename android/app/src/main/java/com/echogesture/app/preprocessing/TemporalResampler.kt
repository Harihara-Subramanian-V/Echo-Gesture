package com.echogesture.app.preprocessing

import kotlin.math.round

/**
 * Native Kotlin Temporal Resampler for EchoGesture Phase 7B.3.
 *
 * Ports `resample_sequence`, `resample_hand_landmarks`, and `resample_interaction_features`
 * from `scripts/prepare_sequences.py` with 100% bitwise parity against offline Python
 * and Stage 2 test fixtures (`tests/fixtures/android_parity/stage2_temporal.json`).
 *
 * Vector Layout (134-D):
 * - [0:63]   : Viewer-Left Hand (Slot 0) 21 normalized landmarks (x, y, z)
 * - [63:126] : Viewer-Right Hand (Slot 1) 21 normalized landmarks (x, y, z)
 * - [126]    : Left Hand Presence flag (1.0f or 0.0f)
 * - [127]    : Right Hand Presence flag (1.0f or 0.0f)
 * - [128]    : Left Hand confidence score (in [0.0, 1.0])
 * - [129]    : Right Hand confidence score (in [0.0, 1.0])
 * - [130]    : Both Hands Co-presence flag (1.0f if presLeft > 0.5f && presRight > 0.5f, else 0.0f)
 * - [131]    : Aspect-ratio corrected diagonal-normalized inter-wrist Euclidean distance
 * - [132:134]: Aspect-ratio corrected diagonal-normalized inter-wrist offsets (dx, dy)
 *
 * Mathematical Rules:
 * 1. Target Grid:
 *    If T <= 1: target_t = [0.0f, ..., 0.0f]
 *    If T > 1: target_t[j] = j * (T - 1).toFloat() / (targetLen - 1).toFloat()
 * 2. Nearest Index:
 *    nearest_idx[j] = kotlin.math.round(target_t[j]).toInt().coerceIn(0, T - 1)
 *    (IEEE-754 round-to-nearest-even matching np.round)
 * 3. Presence Sampling:
 *    target_pres[j] = if (sequence[nearest_idx[j]][pres_idx] > 0.5f) 1.0f else 0.0f
 * 4. Trajectory Interpolation:
 *    Active frames: sequence[t][pres_idx] > 0.5f.
 *    - 0 active frames: coordinates and score remain strictly 0.0f.
 *    - 1 active frame: propagated to all target frames where target_pres[j] > 0.5f.
 *    - >= 2 active frames: linear 1D interpolation in double precision along active frames.
 *      Clamped to 0.0f strictly where target_pres[j] <= 0.5f.
 * 5. Interaction Features:
 *    target_both[j] = if (target_pres_L[j] > 0.5f && target_pres_R[j] > 0.5f) 1.0f else 0.0f
 *    Active both frames: sequence[t][130] > 0.5f.
 *    - 0 active: dist and offsets are 0.0f.
 *    - 1 active: propagated to frames where target_both[j] > 0.5f.
 *    - >= 2 active: linear interpolation in double precision. Clamped to 0.0f where target_both[j] <= 0.5f.
 */
class TemporalResampler(
    private val defaultTargetLen: Int = DEFAULT_TARGET_LEN
) {
    data class ResampledHand(
        val coords: Array<FloatArray>,
        val presence: FloatArray,
        val score: FloatArray
    )

    data class ResampledInteraction(
        val bothPresent: FloatArray,
        val distance: FloatArray,
        val offset: Array<FloatArray>
    )

    companion object {
        const val DEFAULT_TARGET_LEN = 64
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

        /**
         * 1D Linear Interpolation in double precision, matching NumPy's np.interp.
         *
         * @param x Evaluation point
         * @param xp Monotonically increasing x-coordinates of data points
         * @param fp Corresponding y-values of data points
         * @return Interpolated float value
         */
        fun interp(x: Float, xp: FloatArray, fp: FloatArray): Float {
            val n = xp.size
            if (n == 0) return 0.0f
            if (n == 1) return fp[0]
            if (x <= xp[0]) return fp[0]
            if (x >= xp[n - 1]) return fp[n - 1]

            var low = 0
            var high = n - 2
            var i = 0
            while (low <= high) {
                val mid = (low + high) ushr 1
                if (xp[mid] <= x) {
                    i = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }

            val x0 = xp[i].toDouble()
            val x1 = xp[i + 1].toDouble()
            if (x1 == x0) return fp[i]
            val f0 = fp[i].toDouble()
            val f1 = fp[i + 1].toDouble()
            val dydx = (f1 - f0) / (x1 - x0)
            val y = f0 + (x.toDouble() - x0) * dydx
            return y.toFloat()
        }

        /**
         * Resample a single hand's landmarks and presence temporally.
         *
         * @param flatFeats Variable-length sequence of 134-D frames [T][134]
         * @param handIdx 0 for Left Hand (0..62), 1 for Right Hand (63..125)
         * @param targetLen Target sequence length (default: 64)
         */
        fun resampleHandLandmarks(
            flatFeats: Array<FloatArray>,
            handIdx: Int,
            targetLen: Int = DEFAULT_TARGET_LEN
        ): ResampledHand {
            require(targetLen >= 0) { "targetLen must be non-negative: $targetLen" }
            if (targetLen == 0) {
                return ResampledHand(
                    coords = emptyArray(),
                    presence = FloatArray(0),
                    score = FloatArray(0)
                )
            }

            val resCoords = Array(targetLen) { FloatArray(COORDS_PER_HAND) }
            val resPres = FloatArray(targetLen)
            val resScore = FloatArray(targetLen)

            val tCount = flatFeats.size
            if (tCount == 0) {
                return ResampledHand(resCoords, resPres, resScore)
            }

            val coordStart = if (handIdx == 0) SLOT0_START else SLOT1_START
            val presIdx = if (handIdx == 0) INDEX_PRES_LEFT else INDEX_PRES_RIGHT
            val scoreIdx = if (handIdx == 0) INDEX_SCORE_LEFT else INDEX_SCORE_RIGHT

            val targetT = FloatArray(targetLen) { j ->
                if (tCount <= 1 || targetLen <= 1) 0.0f
                else j.toFloat() * (tCount - 1).toFloat() / (targetLen - 1).toFloat()
            }

            // Nearest-neighbor presence
            for (j in 0 until targetLen) {
                val nearest = round(targetT[j]).toInt().coerceIn(0, tCount - 1)
                resPres[j] = if (flatFeats[nearest][presIdx] > 0.5f) 1.0f else 0.0f
            }

            // Collect active frames
            var activeCount = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][presIdx] > 0.5f) activeCount++
            }

            if (activeCount == 0) {
                return ResampledHand(resCoords, resPres, resScore)
            }

            val activeT = FloatArray(activeCount)
            val activeOrig = IntArray(activeCount)
            var aIdx = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][presIdx] > 0.5f) {
                    activeT[aIdx] = t.toFloat()
                    activeOrig[aIdx] = t
                    aIdx++
                }
            }

            if (activeCount == 1) {
                val singleOrig = activeOrig[0]
                val singleRow = flatFeats[singleOrig]
                val sScore = singleRow[scoreIdx]
                for (j in 0 until targetLen) {
                    if (resPres[j] > 0.5f) {
                        System.arraycopy(singleRow, coordStart, resCoords[j], 0, COORDS_PER_HAND)
                        resScore[j] = sScore
                    }
                }
                return ResampledHand(resCoords, resPres, resScore)
            }

            // Multiple active detections (activeCount >= 2)
            for (j in 0 until targetLen) {
                if (resPres[j] > 0.5f) {
                    val x = targetT[j]
                    if (x <= activeT[0]) {
                        val row = flatFeats[activeOrig[0]]
                        System.arraycopy(row, coordStart, resCoords[j], 0, COORDS_PER_HAND)
                        resScore[j] = row[scoreIdx]
                    } else if (x >= activeT[activeCount - 1]) {
                        val row = flatFeats[activeOrig[activeCount - 1]]
                        System.arraycopy(row, coordStart, resCoords[j], 0, COORDS_PER_HAND)
                        resScore[j] = row[scoreIdx]
                    } else {
                        var low = 0
                        var high = activeCount - 2
                        var i = 0
                        while (low <= high) {
                            val mid = (low + high) ushr 1
                            if (activeT[mid] <= x) {
                                i = mid
                                low = mid + 1
                            } else {
                                high = mid - 1
                            }
                        }
                        val x0 = activeT[i].toDouble()
                        val x1 = activeT[i + 1].toDouble()
                        val dx = x1 - x0
                        val dt = x.toDouble() - x0
                        val row0 = flatFeats[activeOrig[i]]
                        val row1 = flatFeats[activeOrig[i + 1]]

                        if (dx == 0.0) {
                            System.arraycopy(row0, coordStart, resCoords[j], 0, COORDS_PER_HAND)
                            resScore[j] = row0[scoreIdx]
                        } else {
                            for (c in 0 until COORDS_PER_HAND) {
                                val f0 = row0[coordStart + c].toDouble()
                                val f1 = row1[coordStart + c].toDouble()
                                val dydx = (f1 - f0) / dx
                                resCoords[j][c] = (f0 + dt * dydx).toFloat()
                            }
                            val s0 = row0[scoreIdx].toDouble()
                            val s1 = row1[scoreIdx].toDouble()
                            val ds = (s1 - s0) / dx
                            resScore[j] = (s0 + dt * ds).toFloat()
                        }
                    }
                }
            }

            return ResampledHand(resCoords, resPres, resScore)
        }

        /**
         * Resample two-hand interaction features: both_hands_present, distance, and offset.
         *
         * @param flatFeats Variable-length sequence of 134-D frames [T][134]
         * @param targetPresLeft Target presence flags for left hand [targetLen]
         * @param targetPresRight Target presence flags for right hand [targetLen]
         * @param targetLen Target sequence length (default: 64)
         */
        fun resampleInteractionFeatures(
            flatFeats: Array<FloatArray>,
            targetPresLeft: FloatArray,
            targetPresRight: FloatArray,
            targetLen: Int = DEFAULT_TARGET_LEN
        ): ResampledInteraction {
            require(targetLen >= 0) { "targetLen must be non-negative: $targetLen" }
            if (targetLen == 0) {
                return ResampledInteraction(
                    bothPresent = FloatArray(0),
                    distance = FloatArray(0),
                    offset = emptyArray()
                )
            }

            val targetBoth = FloatArray(targetLen) { j ->
                if (targetPresLeft[j] > 0.5f && targetPresRight[j] > 0.5f) 1.0f else 0.0f
            }
            val resDist = FloatArray(targetLen)
            val resOffset = Array(targetLen) { FloatArray(2) }

            val tCount = flatFeats.size
            if (tCount == 0) {
                return ResampledInteraction(targetBoth, resDist, resOffset)
            }

            val targetT = FloatArray(targetLen) { j ->
                if (tCount <= 1 || targetLen <= 1) 0.0f
                else j.toFloat() * (tCount - 1).toFloat() / (targetLen - 1).toFloat()
            }

            var activeCount = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][INDEX_BOTH_PRESENT] > 0.5f) activeCount++
            }

            if (activeCount == 0) {
                return ResampledInteraction(targetBoth, resDist, resOffset)
            }

            val activeT = FloatArray(activeCount)
            val activeOrig = IntArray(activeCount)
            var aIdx = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][INDEX_BOTH_PRESENT] > 0.5f) {
                    activeT[aIdx] = t.toFloat()
                    activeOrig[aIdx] = t
                    aIdx++
                }
            }

            if (activeCount == 1) {
                val singleRow = flatFeats[activeOrig[0]]
                val sDist = singleRow[INDEX_INTER_DIST]
                val sDx = singleRow[INDEX_INTER_OFFSET_DX]
                val sDy = singleRow[INDEX_INTER_OFFSET_DY]
                for (j in 0 until targetLen) {
                    if (targetBoth[j] > 0.5f) {
                        resDist[j] = sDist
                        resOffset[j][0] = sDx
                        resOffset[j][1] = sDy
                    }
                }
                return ResampledInteraction(targetBoth, resDist, resOffset)
            }

            // Multiple active detections (activeCount >= 2)
            for (j in 0 until targetLen) {
                if (targetBoth[j] > 0.5f) {
                    val x = targetT[j]
                    if (x <= activeT[0]) {
                        val row = flatFeats[activeOrig[0]]
                        resDist[j] = row[INDEX_INTER_DIST]
                        resOffset[j][0] = row[INDEX_INTER_OFFSET_DX]
                        resOffset[j][1] = row[INDEX_INTER_OFFSET_DY]
                    } else if (x >= activeT[activeCount - 1]) {
                        val row = flatFeats[activeOrig[activeCount - 1]]
                        resDist[j] = row[INDEX_INTER_DIST]
                        resOffset[j][0] = row[INDEX_INTER_OFFSET_DX]
                        resOffset[j][1] = row[INDEX_INTER_OFFSET_DY]
                    } else {
                        var low = 0
                        var high = activeCount - 2
                        var i = 0
                        while (low <= high) {
                            val mid = (low + high) ushr 1
                            if (activeT[mid] <= x) {
                                i = mid
                                low = mid + 1
                            } else {
                                high = mid - 1
                            }
                        }
                        val x0 = activeT[i].toDouble()
                        val x1 = activeT[i + 1].toDouble()
                        val dx = x1 - x0
                        val dt = x.toDouble() - x0
                        val row0 = flatFeats[activeOrig[i]]
                        val row1 = flatFeats[activeOrig[i + 1]]

                        if (dx == 0.0) {
                            resDist[j] = row0[INDEX_INTER_DIST]
                            resOffset[j][0] = row0[INDEX_INTER_OFFSET_DX]
                            resOffset[j][1] = row0[INDEX_INTER_OFFSET_DY]
                        } else {
                            val d0 = row0[INDEX_INTER_DIST].toDouble()
                            val d1 = row1[INDEX_INTER_DIST].toDouble()
                            resDist[j] = (d0 + dt * ((d1 - d0) / dx)).toFloat()

                            val ox0 = row0[INDEX_INTER_OFFSET_DX].toDouble()
                            val ox1 = row1[INDEX_INTER_OFFSET_DX].toDouble()
                            resOffset[j][0] = (ox0 + dt * ((ox1 - ox0) / dx)).toFloat()

                            val oy0 = row0[INDEX_INTER_OFFSET_DY].toDouble()
                            val oy1 = row1[INDEX_INTER_OFFSET_DY].toDouble()
                            resOffset[j][1] = (oy0 + dt * ((oy1 - oy0) / dx)).toFloat()
                        }
                    }
                }
            }

            return ResampledInteraction(targetBoth, resDist, resOffset)
        }

        /**
         * Resample a full (T, 134) sequence into a (target_len, 134) sequence with strict masking.
         *
         * @param flatFeats Variable-length sequence of 134-D frames [T][134]
         * @param targetLen Target sequence length (default: 64)
         * @param destination Optional pre-allocated destination buffer to avoid GC churn
         */
        fun resampleSequence(
            flatFeats: Array<FloatArray>,
            targetLen: Int = DEFAULT_TARGET_LEN,
            destination: Array<FloatArray>? = null
        ): Array<FloatArray> {
            require(targetLen >= 0) { "targetLen must be non-negative: $targetLen" }
            val out = destination ?: Array(targetLen) { FloatArray(FEATURE_DIM) }
            if (targetLen == 0) return out

            require(out.size >= targetLen) {
                "destination size (${out.size}) must be >= targetLen ($targetLen)"
            }

            val tCount = flatFeats.size
            if (tCount == 0) {
                for (j in 0 until targetLen) {
                    out[j].fill(0.0f)
                }
                return out
            }

            // Target grid
            val targetT = FloatArray(targetLen) { j ->
                if (tCount <= 1 || targetLen <= 1) 0.0f
                else j.toFloat() * (tCount - 1).toFloat() / (targetLen - 1).toFloat()
            }

            // Nearest-neighbor presence for left and right hands
            val presLeft = FloatArray(targetLen)
            val presRight = FloatArray(targetLen)
            for (j in 0 until targetLen) {
                val nearest = round(targetT[j]).toInt().coerceIn(0, tCount - 1)
                presLeft[j] = if (flatFeats[nearest][INDEX_PRES_LEFT] > 0.5f) 1.0f else 0.0f
                presRight[j] = if (flatFeats[nearest][INDEX_PRES_RIGHT] > 0.5f) 1.0f else 0.0f
            }

            // Clear destination frames to ensure strict zero-masking
            for (j in 0 until targetLen) {
                out[j].fill(0.0f)
                out[j][INDEX_PRES_LEFT] = presLeft[j]
                out[j][INDEX_PRES_RIGHT] = presRight[j]
            }

            // Resample Left Hand (Slot 0)
            resampleSlotInto(
                flatFeats = flatFeats,
                targetT = targetT,
                targetPres = presLeft,
                coordStart = SLOT0_START,
                presIdx = INDEX_PRES_LEFT,
                scoreIdx = INDEX_SCORE_LEFT,
                targetLen = targetLen,
                out = out
            )

            // Resample Right Hand (Slot 1)
            resampleSlotInto(
                flatFeats = flatFeats,
                targetT = targetT,
                targetPres = presRight,
                coordStart = SLOT1_START,
                presIdx = INDEX_PRES_RIGHT,
                scoreIdx = INDEX_SCORE_RIGHT,
                targetLen = targetLen,
                out = out
            )

            // Interaction features
            resampleInteractionInto(
                flatFeats = flatFeats,
                targetT = targetT,
                presLeft = presLeft,
                presRight = presRight,
                targetLen = targetLen,
                out = out
            )

            return out
        }

        private fun resampleSlotInto(
            flatFeats: Array<FloatArray>,
            targetT: FloatArray,
            targetPres: FloatArray,
            coordStart: Int,
            presIdx: Int,
            scoreIdx: Int,
            targetLen: Int,
            out: Array<FloatArray>
        ) {
            val tCount = flatFeats.size
            var activeCount = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][presIdx] > 0.5f) activeCount++
            }

            if (activeCount == 0) {
                return
            }

            val activeT = FloatArray(activeCount)
            val activeOrig = IntArray(activeCount)
            var aIdx = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][presIdx] > 0.5f) {
                    activeT[aIdx] = t.toFloat()
                    activeOrig[aIdx] = t
                    aIdx++
                }
            }

            if (activeCount == 1) {
                val singleRow = flatFeats[activeOrig[0]]
                val sScore = singleRow[scoreIdx]
                for (j in 0 until targetLen) {
                    if (targetPres[j] > 0.5f) {
                        System.arraycopy(singleRow, coordStart, out[j], coordStart, COORDS_PER_HAND)
                        out[j][scoreIdx] = sScore
                    }
                }
                return
            }

            // Multiple active detections (activeCount >= 2)
            for (j in 0 until targetLen) {
                if (targetPres[j] > 0.5f) {
                    val x = targetT[j]
                    if (x <= activeT[0]) {
                        val row = flatFeats[activeOrig[0]]
                        System.arraycopy(row, coordStart, out[j], coordStart, COORDS_PER_HAND)
                        out[j][scoreIdx] = row[scoreIdx]
                    } else if (x >= activeT[activeCount - 1]) {
                        val row = flatFeats[activeOrig[activeCount - 1]]
                        System.arraycopy(row, coordStart, out[j], coordStart, COORDS_PER_HAND)
                        out[j][scoreIdx] = row[scoreIdx]
                    } else {
                        var low = 0
                        var high = activeCount - 2
                        var i = 0
                        while (low <= high) {
                            val mid = (low + high) ushr 1
                            if (activeT[mid] <= x) {
                                i = mid
                                low = mid + 1
                            } else {
                                high = mid - 1
                            }
                        }
                        val x0 = activeT[i].toDouble()
                        val x1 = activeT[i + 1].toDouble()
                        val dx = x1 - x0
                        val dt = x.toDouble() - x0
                        val row0 = flatFeats[activeOrig[i]]
                        val row1 = flatFeats[activeOrig[i + 1]]

                        if (dx == 0.0) {
                            System.arraycopy(row0, coordStart, out[j], coordStart, COORDS_PER_HAND)
                            out[j][scoreIdx] = row0[scoreIdx]
                        } else {
                            for (c in 0 until COORDS_PER_HAND) {
                                val f0 = row0[coordStart + c].toDouble()
                                val f1 = row1[coordStart + c].toDouble()
                                val dydx = (f1 - f0) / dx
                                out[j][coordStart + c] = (f0 + dt * dydx).toFloat()
                            }
                            val s0 = row0[scoreIdx].toDouble()
                            val s1 = row1[scoreIdx].toDouble()
                            val ds = (s1 - s0) / dx
                            out[j][scoreIdx] = (s0 + dt * ds).toFloat()
                        }
                    }
                }
            }
        }

        private fun resampleInteractionInto(
            flatFeats: Array<FloatArray>,
            targetT: FloatArray,
            presLeft: FloatArray,
            presRight: FloatArray,
            targetLen: Int,
            out: Array<FloatArray>
        ) {
            val tCount = flatFeats.size
            val targetBoth = BooleanArray(targetLen) { j ->
                presLeft[j] > 0.5f && presRight[j] > 0.5f
            }

            for (j in 0 until targetLen) {
                out[j][INDEX_BOTH_PRESENT] = if (targetBoth[j]) 1.0f else 0.0f
            }

            var activeCount = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][INDEX_BOTH_PRESENT] > 0.5f) activeCount++
            }

            if (activeCount == 0) {
                return
            }

            val activeT = FloatArray(activeCount)
            val activeOrig = IntArray(activeCount)
            var aIdx = 0
            for (t in 0 until tCount) {
                if (flatFeats[t][INDEX_BOTH_PRESENT] > 0.5f) {
                    activeT[aIdx] = t.toFloat()
                    activeOrig[aIdx] = t
                    aIdx++
                }
            }

            if (activeCount == 1) {
                val singleRow = flatFeats[activeOrig[0]]
                val sDist = singleRow[INDEX_INTER_DIST]
                val sDx = singleRow[INDEX_INTER_OFFSET_DX]
                val sDy = singleRow[INDEX_INTER_OFFSET_DY]
                for (j in 0 until targetLen) {
                    if (targetBoth[j]) {
                        out[j][INDEX_INTER_DIST] = sDist
                        out[j][INDEX_INTER_OFFSET_DX] = sDx
                        out[j][INDEX_INTER_OFFSET_DY] = sDy
                    }
                }
                return
            }

            // Multiple active detections (activeCount >= 2)
            for (j in 0 until targetLen) {
                if (targetBoth[j]) {
                    val x = targetT[j]
                    if (x <= activeT[0]) {
                        val row = flatFeats[activeOrig[0]]
                        out[j][INDEX_INTER_DIST] = row[INDEX_INTER_DIST]
                        out[j][INDEX_INTER_OFFSET_DX] = row[INDEX_INTER_OFFSET_DX]
                        out[j][INDEX_INTER_OFFSET_DY] = row[INDEX_INTER_OFFSET_DY]
                    } else if (x >= activeT[activeCount - 1]) {
                        val row = flatFeats[activeOrig[activeCount - 1]]
                        out[j][INDEX_INTER_DIST] = row[INDEX_INTER_DIST]
                        out[j][INDEX_INTER_OFFSET_DX] = row[INDEX_INTER_OFFSET_DX]
                        out[j][INDEX_INTER_OFFSET_DY] = row[INDEX_INTER_OFFSET_DY]
                    } else {
                        var low = 0
                        var high = activeCount - 2
                        var i = 0
                        while (low <= high) {
                            val mid = (low + high) ushr 1
                            if (activeT[mid] <= x) {
                                i = mid
                                low = mid + 1
                            } else {
                                high = mid - 1
                            }
                        }
                        val x0 = activeT[i].toDouble()
                        val x1 = activeT[i + 1].toDouble()
                        val dx = x1 - x0
                        val dt = x.toDouble() - x0
                        val row0 = flatFeats[activeOrig[i]]
                        val row1 = flatFeats[activeOrig[i + 1]]

                        if (dx == 0.0) {
                            out[j][INDEX_INTER_DIST] = row0[INDEX_INTER_DIST]
                            out[j][INDEX_INTER_OFFSET_DX] = row0[INDEX_INTER_OFFSET_DX]
                            out[j][INDEX_INTER_OFFSET_DY] = row0[INDEX_INTER_OFFSET_DY]
                        } else {
                            val d0 = row0[INDEX_INTER_DIST].toDouble()
                            val d1 = row1[INDEX_INTER_DIST].toDouble()
                            out[j][INDEX_INTER_DIST] = (d0 + dt * ((d1 - d0) / dx)).toFloat()

                            val ox0 = row0[INDEX_INTER_OFFSET_DX].toDouble()
                            val ox1 = row1[INDEX_INTER_OFFSET_DX].toDouble()
                            out[j][INDEX_INTER_OFFSET_DX] = (ox0 + dt * ((ox1 - ox0) / dx)).toFloat()

                            val oy0 = row0[INDEX_INTER_OFFSET_DY].toDouble()
                            val oy1 = row1[INDEX_INTER_OFFSET_DY].toDouble()
                            out[j][INDEX_INTER_OFFSET_DY] = (oy0 + dt * ((oy1 - oy0) / dx)).toFloat()
                        }
                    }
                }
            }
        }

        fun resample(
            sequence: Array<FloatArray>,
            targetLen: Int = DEFAULT_TARGET_LEN,
            destination: Array<FloatArray>? = null
        ): Array<FloatArray> = resampleSequence(sequence, targetLen, destination)

        fun resample_sequence(
            sequence: Array<FloatArray>,
            targetLen: Int = DEFAULT_TARGET_LEN,
            destination: Array<FloatArray>? = null
        ): Array<FloatArray> = resampleSequence(sequence, targetLen, destination)

        fun resample_hand_landmarks(
            sequence: Array<FloatArray>,
            handIdx: Int,
            targetLen: Int = DEFAULT_TARGET_LEN
        ): ResampledHand = resampleHandLandmarks(sequence, handIdx, targetLen)

        fun resample_interaction_features(
            sequence: Array<FloatArray>,
            targetPresLeft: FloatArray,
            targetPresRight: FloatArray,
            targetLen: Int = DEFAULT_TARGET_LEN
        ): ResampledInteraction = resampleInteractionFeatures(sequence, targetPresLeft, targetPresRight, targetLen)
    }

    /**
     * Resample a sequence using instance configuration.
     *
     * @param sequence Variable-length sequence of 134-D frames [T][134]
     * @param targetLen Target sequence length (default: defaultTargetLen = 64)
     * @param destination Optional pre-allocated destination buffer to avoid GC churn
     */
    fun resample(
        sequence: Array<FloatArray>,
        targetLen: Int = defaultTargetLen,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> = resampleSequence(sequence, targetLen, destination)

    fun resample_sequence(
        sequence: Array<FloatArray>,
        targetLen: Int = defaultTargetLen,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> = resampleSequence(sequence, targetLen, destination)

    fun resample_hand_landmarks(
        sequence: Array<FloatArray>,
        handIdx: Int,
        targetLen: Int = defaultTargetLen
    ): ResampledHand = resampleHandLandmarks(sequence, handIdx, targetLen)

    fun resample_interaction_features(
        sequence: Array<FloatArray>,
        targetPresLeft: FloatArray,
        targetPresRight: FloatArray,
        targetLen: Int = defaultTargetLen
    ): ResampledInteraction = resampleInteractionFeatures(sequence, targetPresLeft, targetPresRight, targetLen)
}
