package com.echogesture.app.inference

import android.content.Context
import com.echogesture.app.preprocessing.SequencePreprocessor263D
import java.io.File
import java.nio.ByteBuffer

/**
 * Granular latency breakdown and prediction result for [EchoGestureInferenceEngine].
 */
data class DetailedInferenceResult(
    val predictionResult: PredictionResult,
    val preprocessMs: Double = 0.0,
    val quantizeMs: Double = 0.0,
    val modelMs: Double = 0.0,
    val decodeMs: Double = 0.0,
    val totalInferenceMs: Double = 0.0
)

/**
 * High-level unified inference engine for EchoGesture.
 *
 * Connects the complete on-device pipeline:
 * ```
 * Raw 134-D Frames [T, 134]
 *         │
 *         ▼ (Temporal Resampling L=64 + Active Z-Score + 129-D Velocity)
 * SequencePreprocessor263D
 *         │
 *         ▼ FloatArray(16,832) [64, 263]
 * Int8Quantizer
 *         │
 *         ▼ DirectByteBuffer(16,832) INT8
 * LiteRtModelRunner (LiteRT / TFLite INT8 Graph)
 *         │
 *         ▼ DirectByteBuffer(40) INT8
 * PredictionDecoder
 *         │
 *         ▼
 * PredictionResult (Top-1, Top-5, Confidences)
 * ```
 *
 * Designed for zero-allocation execution on the inference hot path.
 */
class EchoGestureInferenceEngine(
    val modelRunner: LiteRtModelRunner,
    val preprocessor: SequencePreprocessor263D? = null,
    val quantizer: Int8Quantizer = Int8Quantizer(),
    val decoder: PredictionDecoder = PredictionDecoder()
) : AutoCloseable {

    // Preallocated flat buffer for zero GC allocations during inference
    private val flatFeatureBuffer: FloatArray = FloatArray(SequencePreprocessor263D.TOTAL_ELEMENTS)
    private val rawOutputBytes: ByteArray = ByteArray(ModelContract.NUM_CLASSES)

    /**
     * Executes end-to-end inference from variable-length raw 134-D frames.
     *
     * @param rawSequence Variable length input sequence [T, 134].
     * @return Formatted [PredictionResult] with Top-1 and Top-5 class decisions.
     */
    fun predict(rawSequence: Array<FloatArray>): PredictionResult {
        return predictDetailed(rawSequence).predictionResult
    }

    /**
     * Executes end-to-end inference from variable-length raw 134-D frames with granular stage timing.
     */
    fun predictDetailed(rawSequence: Array<FloatArray>): DetailedInferenceResult {
        val prep = preprocessor ?: throw IllegalStateException("SequencePreprocessor263D is required for raw sequence inference")
        val tPre0 = System.nanoTime()
        prep.processToFlat(rawSequence, flatFeatureBuffer)
        val tPre1 = System.nanoTime()
        val prepMs = (tPre1 - tPre0) / 1_000_000.0

        return predictPreprocessedDetailed(flatFeatureBuffer, prepMs)
    }

    /**
     * Executes inference on already preprocessed 2D features of shape [64, 263].
     */
    fun predictPreprocessed(preprocessed263: Array<FloatArray>): PredictionResult {
        require(preprocessed263.size == ModelContract.SEQUENCE_LENGTH) {
            "Expected sequence length ${ModelContract.SEQUENCE_LENGTH}, got ${preprocessed263.size}"
        }
        var offset = 0
        for (t in 0 until ModelContract.SEQUENCE_LENGTH) {
            require(preprocessed263[t].size == ModelContract.FEATURE_DIM) {
                "Frame $t dimension ${preprocessed263[t].size} != ${ModelContract.FEATURE_DIM}"
            }
            System.arraycopy(preprocessed263[t], 0, flatFeatureBuffer, offset, ModelContract.FEATURE_DIM)
            offset += ModelContract.FEATURE_DIM
        }
        return predictPreprocessed(flatFeatureBuffer)
    }

    /**
     * Executes inference on already preprocessed flat float array of length 16,832.
     */
    fun predictPreprocessed(flatFeatures16832: FloatArray): PredictionResult {
        return predictPreprocessedDetailed(flatFeatures16832, 0.0).predictionResult
    }

    /**
     * Executes inference on already preprocessed flat float array with granular stage timing.
     */
    fun predictPreprocessedDetailed(flatFeatures16832: FloatArray, prepMs: Double = 0.0): DetailedInferenceResult {
        require(flatFeatures16832.size == ModelContract.INPUT_BUFFER_SIZE_BYTES) {
            "Expected ${ModelContract.INPUT_BUFFER_SIZE_BYTES} floats, got ${flatFeatures16832.size}"
        }

        val tTotal0 = System.nanoTime()

        // 1. Quantize directly into LiteRtModelRunner's direct input buffer
        val tQuant0 = System.nanoTime()
        quantizer.quantize(flatFeatures16832, modelRunner.inputBuffer)
        val tQuant1 = System.nanoTime()

        // 2. Execute INT8 model graph
        val tModel0 = System.nanoTime()
        modelRunner.runInference(modelRunner.inputBuffer, modelRunner.outputBuffer)
        val tModel1 = System.nanoTime()

        // 3. Decode Top-5 predictions from output buffer
        val tDec0 = System.nanoTime()
        val result = decoder.decode(modelRunner.outputBuffer, modelRunner.getClassLabels())
        val tDec1 = System.nanoTime()

        val tTotal1 = System.nanoTime()

        val quantizeMs = (tQuant1 - tQuant0) / 1_000_000.0
        val modelMs = (tModel1 - tModel0) / 1_000_000.0
        val decodeMs = (tDec1 - tDec0) / 1_000_000.0
        val totalMs = prepMs + (tTotal1 - tTotal0) / 1_000_000.0

        return DetailedInferenceResult(
            predictionResult = result,
            preprocessMs = prepMs,
            quantizeMs = quantizeMs,
            modelMs = modelMs,
            decodeMs = decodeMs,
            totalInferenceMs = totalMs
        )
    }

    /**
     * Executes inference on pre-quantized INT8 byte array of length 16,832.
     */
    fun predictQuantized(quantizedBytes: ByteArray): PredictionResult {
        require(quantizedBytes.size == ModelContract.INPUT_BUFFER_SIZE_BYTES) {
            "Expected ${ModelContract.INPUT_BUFFER_SIZE_BYTES} bytes, got ${quantizedBytes.size}"
        }

        modelRunner.inputBuffer.rewind()
        modelRunner.inputBuffer.put(quantizedBytes)
        modelRunner.inputBuffer.rewind()

        modelRunner.runInference(modelRunner.inputBuffer, modelRunner.outputBuffer)

        return decoder.decode(modelRunner.outputBuffer, modelRunner.getClassLabels())
    }

    override fun close() {
        modelRunner.close()
    }

    companion object {
        /**
         * Creates an engine from Android application assets.
         */
        fun fromAssets(
            context: Context,
            modelAssetPath: String = ModelContract.MODEL_ASSET_PATH,
            vocabAssetPath: String = ModelContract.LABEL_ENCODER_PATH,
            scalerAssetPath: String = SequencePreprocessor263D.DEFAULT_SCALER_ASSET_PATH,
            numThreads: Int = 2
        ): EchoGestureInferenceEngine {
            val runner = LiteRtModelRunner(context)
            val status = runner.initializeFromAssets(context, modelAssetPath, vocabAssetPath, numThreads)
            check(status.isLoaded) { "Failed to initialize LiteRtModelRunner: ${status.errorMessage}" }

            val preprocessor = SequencePreprocessor263D.fromAssets(context, scalerAssetPath)
            return EchoGestureInferenceEngine(
                modelRunner = runner,
                preprocessor = preprocessor
            )
        }

        /**
         * Creates an engine from files (useful for host JVM testing).
         */
        fun fromFiles(
            modelFile: File,
            vocabFile: File,
            scalerFile: File,
            numThreads: Int = 2
        ): EchoGestureInferenceEngine {
            val runner = LiteRtModelRunner()
            val status = runner.initializeFromFile(modelFile, vocabFile, numThreads)
            check(status.isLoaded) { "Failed to initialize LiteRtModelRunner from files: ${status.errorMessage}" }

            val preprocessor = SequencePreprocessor263D.fromFile(scalerFile)
            return EchoGestureInferenceEngine(
                modelRunner = runner,
                preprocessor = preprocessor
            )
        }

        /**
         * Creates an engine from in-memory buffers and strings.
         */
        fun fromBuffers(
            modelBuffer: ByteBuffer,
            vocabJson: String,
            scalerJson: String,
            numThreads: Int = 2
        ): EchoGestureInferenceEngine {
            val runner = LiteRtModelRunner()
            val status = runner.initializeFromBuffer(modelBuffer, vocabJson, numThreads)
            check(status.isLoaded) { "Failed to initialize LiteRtModelRunner from buffer: ${status.errorMessage}" }

            val preprocessor = SequencePreprocessor263D.fromJson(scalerJson)
            return EchoGestureInferenceEngine(
                modelRunner = runner,
                preprocessor = preprocessor
            )
        }
    }
}
