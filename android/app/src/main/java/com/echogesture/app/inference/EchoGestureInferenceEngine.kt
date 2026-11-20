package com.echogesture.app.inference

import android.content.Context
import com.echogesture.app.preprocessing.SequencePreprocessor263D
import java.io.File
import java.nio.ByteBuffer

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
        val prep = preprocessor ?: throw IllegalStateException("SequencePreprocessor263D is required for raw sequence inference")
        prep.processToFlat(rawSequence, flatFeatureBuffer)
        return predictPreprocessed(flatFeatureBuffer)
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
        require(flatFeatures16832.size == ModelContract.INPUT_BUFFER_SIZE_BYTES) {
            "Expected ${ModelContract.INPUT_BUFFER_SIZE_BYTES} floats, got ${flatFeatures16832.size}"
        }

        // Quantize directly into LiteRtModelRunner's direct input buffer
        quantizer.quantize(flatFeatures16832, modelRunner.inputBuffer)

        // Execute INT8 model graph
        modelRunner.runInference(modelRunner.inputBuffer, modelRunner.outputBuffer)

        // Decode Top-5 predictions from output buffer
        return decoder.decode(modelRunner.outputBuffer, modelRunner.getClassLabels())
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
