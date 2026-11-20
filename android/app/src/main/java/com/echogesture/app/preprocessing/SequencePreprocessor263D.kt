package com.echogesture.app.preprocessing

import android.content.Context
import java.io.File

/**
 * Fused End-to-End Stage 2 Preprocessing Pipeline for EchoGesture Phase 7B.3.
 *
 * Integrates:
 * 1. [TemporalResampler]: Dynamic temporal resampling from variable length T to fixed L=64 frames.
 * 2. [SequenceNormalizer]: Active-frame Z-score standardization using frozen dataset parameters.
 * 3. [VelocityFeatureExtractor]: 129-D first-order temporal velocity extraction with transition spike prevention.
 *
 * Pipeline Flow:
 * ```
 * [T, 134] (Raw Stage 1 Feature Sequence)
 *     │
 *     ▼ TemporalResampler
 * [64, 134] (Fixed-Grid Resampled Sequence)
 *     │
 *     ▼ SequenceNormalizer
 * [64, 134] (Active-Frame Standardized Sequence)
 *     │
 *     ├───► VelocityFeatureExtractor ───► [64, 129] (Spike-Free Velocities)
 *     │                                           │
 *     └───────────────────┬───────────────────────┘
 *                         │ (Concatenation)
 *                         ▼
 *                   [64, 263] (LiteRT Model Input Tensor)
 * ```
 *
 * Output Tensor Structure [64, 263]:
 * - [0..133]   : Standardized 134-D spatial features.
 * - [134..196] : Left hand landmark velocities (63 channels).
 * - [197..259] : Right hand landmark velocities (63 channels).
 * - [260..262] : Two-hand interaction velocities (3 channels).
 */
class SequencePreprocessor263D(
    val normalizer: SequenceNormalizer,
    val resampler: TemporalResampler = TemporalResampler(TARGET_FRAMES),
    val velocityExtractor: VelocityFeatureExtractor = VelocityFeatureExtractor(),
    private val preallocateBuffers: Boolean = true
) {
    // Reusable internal buffers for zero-allocation execution during 30 FPS inference
    private val resampledBuffer: Array<FloatArray>? =
        if (preallocateBuffers) Array(TARGET_FRAMES) { FloatArray(FEATURE_DIM_134) } else null
    private val normalizedBuffer: Array<FloatArray>? =
        if (preallocateBuffers) Array(TARGET_FRAMES) { FloatArray(FEATURE_DIM_134) } else null
    private val velocityBuffer: Array<FloatArray>? =
        if (preallocateBuffers) Array(TARGET_FRAMES) { FloatArray(FEATURE_DIM_129) } else null
    private val outputBuffer: Array<FloatArray>? =
        if (preallocateBuffers) Array(TARGET_FRAMES) { FloatArray(OUTPUT_DIM_263) } else null

    /**
     * Executes the complete Stage 2 preprocessing pipeline.
     *
     * @param rawSequence Variable length input sequence of shape [T, 134].
     * @param destination Optional preallocated destination tensor of shape [64, 263].
     * @return Fully processed 263-D tensor of shape [64, 263].
     */
    fun process(
        rawSequence: Array<FloatArray>,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> {
        val out = destination ?: (outputBuffer ?: Array(TARGET_FRAMES) { FloatArray(OUTPUT_DIM_263) })
        require(out.size == TARGET_FRAMES) {
            "Destination outer dimension must be $TARGET_FRAMES, got ${out.size}"
        }

        // 1. Dynamic Temporal Resampling: [T, 134] -> [64, 134]
        val resampled = resampler.resample(rawSequence, TARGET_FRAMES, resampledBuffer)

        // 2. Active-Frame Z-Score Normalization: [64, 134] -> [64, 134]
        val normalized = normalizer.normalize(resampled, normalizedBuffer)

        // 3. First-Order Temporal Velocity Extraction: [64, 134] -> [64, 129]
        val velocity = velocityExtractor.extract(normalized, velocityBuffer)

        // 4. Feature Concatenation: 134 + 129 = 263
        velocityExtractor.concatenate(normalized, velocity, out)

        return out
    }

    /**
     * Executes the preprocessing pipeline and returns a flattened float array of size [64 * 263].
     *
     * Useful for direct memory copy into LiteRT ByteBuffers.
     *
     * @param rawSequence Variable length input sequence of shape [T, 134].
     * @param flatDestination Optional preallocated flat buffer of size 16,832.
     * @return Flattened FloatArray of length 16,832.
     */
    fun processToFlat(
        rawSequence: Array<FloatArray>,
        flatDestination: FloatArray? = null
    ): FloatArray {
        val flatOut = flatDestination ?: FloatArray(TOTAL_ELEMENTS)
        require(flatOut.size == TOTAL_ELEMENTS) {
            "Flat destination size must be $TOTAL_ELEMENTS, got ${flatOut.size}"
        }

        val processed2D = process(rawSequence)
        var offset = 0
        for (t in 0 until TARGET_FRAMES) {
            System.arraycopy(processed2D[t], 0, flatOut, offset, OUTPUT_DIM_263)
            offset += OUTPUT_DIM_263
        }

        return flatOut
    }

    companion object {
        const val TARGET_FRAMES = 64
        const val FEATURE_DIM_134 = 134
        const val FEATURE_DIM_129 = 129
        const val OUTPUT_DIM_263 = 263
        const val TOTAL_ELEMENTS = TARGET_FRAMES * OUTPUT_DIM_263 // 16,832

        const val DEFAULT_SCALER_ASSET_PATH = "scaler_params.json"

        /**
         * Factory method creating a preprocessor loaded from Android assets.
         */
        fun fromAssets(
            context: Context,
            assetPath: String = DEFAULT_SCALER_ASSET_PATH,
            preallocateBuffers: Boolean = true
        ): SequencePreprocessor263D {
            val normalizer = SequenceNormalizer.fromAssets(context, assetPath)
            return SequencePreprocessor263D(
                normalizer = normalizer,
                preallocateBuffers = preallocateBuffers
            )
        }

        /**
         * Factory method creating a preprocessor from JSON string (used in unit tests / host JVM).
         */
        fun fromJson(
            jsonString: String,
            preallocateBuffers: Boolean = true
        ): SequencePreprocessor263D {
            val normalizer = SequenceNormalizer.fromJson(jsonString)
            return SequencePreprocessor263D(
                normalizer = normalizer,
                preallocateBuffers = preallocateBuffers
            )
        }

        /**
         * Factory method creating a preprocessor from a JSON file.
         */
        fun fromFile(
            file: File,
            preallocateBuffers: Boolean = true
        ): SequencePreprocessor263D {
            val normalizer = SequenceNormalizer.fromFile(file)
            return SequencePreprocessor263D(
                normalizer = normalizer,
                preallocateBuffers = preallocateBuffers
            )
        }
    }
}
