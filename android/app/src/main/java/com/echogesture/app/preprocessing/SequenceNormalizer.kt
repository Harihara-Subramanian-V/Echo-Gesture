package com.echogesture.app.preprocessing

import android.content.Context
import org.json.JSONObject
import java.io.File

/**
 * Standardizes 134-D spatial feature sequences using frozen dataset mean and standard deviation parameters.
 *
 * Implements active-frame Z-score standardization matching Python `LiveTemporalPreprocessor.process_window`
 * and `scripts/prepare_sequences.py:apply_scaler`:
 * - Channels 0..62: Left hand landmarks (normalized only when left hand presence > 0.5f, else clamped to 0.0f).
 * - Channels 63..125: Right hand landmarks (normalized only when right hand presence > 0.5f, else clamped to 0.0f).
 * - Channels 126..130: Unmodified metadata channels (presence flags, handedness scores, both-present flag).
 * - Channels 131..133: Two-hand interaction features (normalized only when both-hands flag > 0.5f, else clamped to 0.0f).
 *
 * Mathematical formula for active channels:
 *     normalized[t][c] = (seq[t][c] - mean[c]) / (std[c] + 1e-6f)
 * Inactive channels are strictly clamped to 0.0f.
 */
class SequenceNormalizer(
    val mean: FloatArray,
    val std: FloatArray
) {
    init {
        require(mean.size == FEATURE_DIM) {
            "Expected mean dimension $FEATURE_DIM, got ${mean.size}"
        }
        require(std.size == FEATURE_DIM) {
            "Expected std dimension $FEATURE_DIM, got ${std.size}"
        }
    }

    /**
     * Standardizes a sequence of 134-D frames.
     *
     * @param resampledSequence Sequence of shape [T][134] (typically T=64).
     * @param destination Optional preallocated destination buffer of shape [T][134] to avoid GC allocations.
     *                    May also be [resampledSequence] itself for in-place normalization.
     * @return Standardized sequence of shape [T][134].
     */
    fun normalize(
        resampledSequence: Array<FloatArray>,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> {
        val tCount = resampledSequence.size
        if (tCount == 0) {
            return destination ?: Array(0) { FloatArray(FEATURE_DIM) }
        }

        val out = destination ?: Array(tCount) { FloatArray(FEATURE_DIM) }
        require(out.size == tCount) {
            "Destination size (${out.size}) does not match sequence size ($tCount)"
        }

        for (t in 0 until tCount) {
            val frame = resampledSequence[t]
            val outFrame = out[t]

            require(frame.size == FEATURE_DIM) {
                "Frame $t feature dimension ${frame.size} does not match expected $FEATURE_DIM"
            }
            require(outFrame.size == FEATURE_DIM) {
                "Destination frame $t dimension ${outFrame.size} does not match expected $FEATURE_DIM"
            }

            val pL = frame[INDEX_LEFT_PRESENT] > PRESENCE_THRESHOLD
            val pR = frame[INDEX_RIGHT_PRESENT] > PRESENCE_THRESHOLD
            val pBoth = frame[INDEX_BOTH_PRESENT] > PRESENCE_THRESHOLD

            // 1. Left Hand Landmarks (channels 0..62)
            if (pL) {
                for (c in 0 until 63) {
                    outFrame[c] = (frame[c] - mean[c]) / (std[c] + EPSILON)
                }
            } else {
                for (c in 0 until 63) {
                    outFrame[c] = 0.0f
                }
            }

            // 2. Right Hand Landmarks (channels 63..125)
            if (pR) {
                for (c in 63 until 126) {
                    outFrame[c] = (frame[c] - mean[c]) / (std[c] + EPSILON)
                }
            } else {
                for (c in 63 until 126) {
                    outFrame[c] = 0.0f
                }
            }

            // 3. Presence flags and scores (channels 126..130) pass through unmodified
            outFrame[126] = frame[126]
            outFrame[127] = frame[127]
            outFrame[128] = frame[128]
            outFrame[129] = frame[129]
            outFrame[130] = frame[130]

            // 4. Two-hand interaction features (channels 131..133)
            if (pBoth) {
                for (c in 131 until 134) {
                    outFrame[c] = (frame[c] - mean[c]) / (std[c] + EPSILON)
                }
            } else {
                for (c in 131 until 134) {
                    outFrame[c] = 0.0f
                }
            }
        }

        return out
    }

    companion object {
        const val FEATURE_DIM = 134
        const val SEQUENCE_LENGTH = 64
        const val EPSILON = 1e-6f
        const val PRESENCE_THRESHOLD = 0.5f

        const val INDEX_LEFT_PRESENT = 126
        const val INDEX_RIGHT_PRESENT = 127
        const val INDEX_LEFT_SCORE = 128
        const val INDEX_RIGHT_SCORE = 129
        const val INDEX_BOTH_PRESENT = 130

        const val DEFAULT_ASSET_PATH = "scaler_params.json"

        /**
         * Parses a SequenceNormalizer from a JSON string representing scaler parameters.
         * Supports root-level {"mean": [...], "std": [...]} as well as nested {"scaler": {"mean": ..., "std": ...}}.
         */
        @JvmStatic
        fun fromJson(jsonString: String): SequenceNormalizer {
            val root = JSONObject(jsonString)
            val scalerObj = if (root.has("scaler")) root.getJSONObject("scaler") else root

            val meanArray = scalerObj.getJSONArray("mean")
            val stdArray = scalerObj.getJSONArray("std")

            require(meanArray.length() == FEATURE_DIM) {
                "Expected mean dimension $FEATURE_DIM, got ${meanArray.length()}"
            }
            require(stdArray.length() == FEATURE_DIM) {
                "Expected std dimension $FEATURE_DIM, got ${stdArray.length()}"
            }

            val mean = FloatArray(FEATURE_DIM)
            val std = FloatArray(FEATURE_DIM)
            for (i in 0 until FEATURE_DIM) {
                mean[i] = meanArray.getDouble(i).toFloat()
                std[i] = stdArray.getDouble(i).toFloat()
            }

            return SequenceNormalizer(mean, std)
        }

        /**
         * Loads scaler parameters from Android assets.
         */
        @JvmStatic
        fun fromAssets(context: Context, assetPath: String = DEFAULT_ASSET_PATH): SequenceNormalizer {
            val jsonString = context.assets.open(assetPath).bufferedReader(Charsets.UTF_8).use { it.readText() }
            return fromJson(jsonString)
        }

        /**
         * Loads scaler parameters directly from a File (useful for testing and standalone execution).
         */
        @JvmStatic
        fun fromFile(file: File): SequenceNormalizer {
            return fromJson(file.readText(Charsets.UTF_8))
        }
    }
}
