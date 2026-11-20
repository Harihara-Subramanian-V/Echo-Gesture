package com.echogesture.app.inference

import java.nio.ByteBuffer

/**
 * Single candidate prediction ranking.
 */
data class PredictionCandidate(
    val rank: Int,
    val classId: Int,
    val className: String,
    val quantizedScore: Byte,
    val confidence: Float
)

/**
 * Structured prediction result containing top-1 decision, top-5 candidates,
 * and the complete dequantized probability distribution across all 40 classes.
 */
data class PredictionResult(
    val topClassId: Int,
    val topClassName: String,
    val topConfidence: Float,
    val topQuantizedScore: Byte,
    val top5: List<PredictionCandidate>,
    val allProbabilities: FloatArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as PredictionResult

        if (topClassId != other.topClassId) return false
        if (topClassName != other.topClassName) return false
        if (topConfidence != other.topConfidence) return false
        if (topQuantizedScore != other.topQuantizedScore) return false
        if (top5 != other.top5) return false
        if (!allProbabilities.contentEquals(other.allProbabilities)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = topClassId
        result = 31 * result + topClassName.hashCode()
        result = 31 * result + topConfidence.hashCode()
        result = 31 * result + topQuantizedScore.hashCode()
        result = 31 * result + top5.hashCode()
        result = 31 * result + allProbabilities.contentHashCode()
        return result
    }
}

/**
 * Prediction decoder for EchoGesture INT8 quantized TFLite model output.
 *
 * Implements dequantization and monotonic argmax ranking matching:
 * - reports/production_inference_contract.md (Stage 9)
 * - scripts/echogesture_runtime.py (Lines 117-147)
 * - ModelContract specification
 *
 * Mathematical Formula:
 *     p = (q - Z_out) * S_out = (q + 128) / 256.0f
 *
 * Because the TFLite model graph includes Softmax prior to output quantization,
 * dequantized values are directly valid probabilities in [0.0, 1.0].
 *
 * Because dequantization is strictly linear and monotonic, sorting by signed int8 Byte
 * in descending order is mathematically identical to sorting by float confidence.
 */
class PredictionDecoder(
    val scale: Float = ModelContract.OUTPUT_SCALE,
    val zeroPoint: Int = ModelContract.OUTPUT_ZERO_POINT
) {
    /**
     * Dequantize a single output byte to a float32 confidence in [0.0, 1.0].
     */
    fun dequantize(byteValue: Byte): Float = Companion.dequantize(byteValue, scale, zeroPoint)

    /**
     * Dequantize all 40 output bytes into a float array of probabilities.
     */
    fun dequantizeAll(outputBytes: ByteArray, destination: FloatArray? = null): FloatArray =
        Companion.dequantizeAll(outputBytes, destination, scale, zeroPoint)

    /**
     * Decode an INT8 output byte array into a structured PredictionResult.
     */
    fun decode(
        outputBytes: ByteArray,
        vocabulary: List<String>,
        topK: Int = 5
    ): PredictionResult = Companion.decode(outputBytes, vocabulary, topK, scale, zeroPoint)

    /**
     * Decode an INT8 output ByteBuffer into a structured PredictionResult.
     */
    fun decode(
        outputBuffer: ByteBuffer,
        vocabulary: List<String>,
        topK: Int = 5
    ): PredictionResult = Companion.decode(outputBuffer, vocabulary, topK, scale, zeroPoint)

    companion object {
        const val OUTPUT_SCALE: Float = ModelContract.OUTPUT_SCALE
        const val OUTPUT_ZERO_POINT: Int = ModelContract.OUTPUT_ZERO_POINT
        const val NUM_CLASSES: Int = ModelContract.NUM_CLASSES
        const val OUTPUT_BUFFER_SIZE_BYTES: Int = ModelContract.OUTPUT_BUFFER_SIZE_BYTES

        /**
         * Dequantize a single signed int8 byte value to a float confidence score in [0.0, 1.0].
         * Formula: p = (q - Z_out) * S_out = (q + 128) / 256.0f
         */
        fun dequantize(
            byteValue: Byte,
            scale: Float = OUTPUT_SCALE,
            zeroPoint: Int = OUTPUT_ZERO_POINT
        ): Float {
            val qInt = byteValue.toInt()
            return ((qInt - zeroPoint) * scale).coerceIn(0.0f, 1.0f)
        }

        /**
         * Dequantize all 40 output bytes into a FloatArray of probabilities in [0.0, 1.0].
         */
        fun dequantizeAll(
            outputBytes: ByteArray,
            destination: FloatArray? = null,
            scale: Float = OUTPUT_SCALE,
            zeroPoint: Int = OUTPUT_ZERO_POINT
        ): FloatArray {
            require(outputBytes.size == NUM_CLASSES) {
                "Expected output array of size $NUM_CLASSES, got ${outputBytes.size}"
            }
            val result = destination ?: FloatArray(NUM_CLASSES)
            require(result.size == NUM_CLASSES) {
                "Destination array size ${result.size} does not match required $NUM_CLASSES"
            }

            for (i in 0 until NUM_CLASSES) {
                result[i] = dequantize(outputBytes[i], scale, zeroPoint)
            }
            return result
        }

        /**
         * Resolves class name from vocabulary, falling back safely to "class_$classId"
         * if vocabulary is empty or smaller than NUM_CLASSES.
         */
        fun resolveClassName(classId: Int, vocabulary: List<String>): String {
            return if (classId in vocabulary.indices) vocabulary[classId] else "class_$classId"
        }

        /**
         * Decode an INT8 output byte array into a structured PredictionResult.
         *
         * @param outputBytes Byte array of exactly 40 signed bytes from TFLite execution.
         * @param vocabulary Ordered list of class label strings.
         * @param topK Number of candidate classes to return in top5 list (clamped to 1..40, default 5).
         */
        fun decode(
            outputBytes: ByteArray,
            vocabulary: List<String>,
            topK: Int = 5,
            scale: Float = OUTPUT_SCALE,
            zeroPoint: Int = OUTPUT_ZERO_POINT
        ): PredictionResult {
            require(outputBytes.size == NUM_CLASSES) {
                "Expected output array of size $NUM_CLASSES, got ${outputBytes.size}"
            }
            val actualK = topK.coerceIn(1, NUM_CLASSES)

            // Monotonic sorting: signed byte ordering (descending) is strictly identical
            // to float confidence ordering. Ties broken stably by lower classId first.
            val sortedIndices = Array(NUM_CLASSES) { it }
            sortedIndices.sortWith { a, b ->
                val cmp = outputBytes[b].compareTo(outputBytes[a])
                if (cmp != 0) cmp else a.compareTo(b)
            }

            val allProbabilities = dequantizeAll(outputBytes, null, scale, zeroPoint)

            val topCandidates = ArrayList<PredictionCandidate>(actualK)
            for (rank in 1..actualK) {
                val cId = sortedIndices[rank - 1]
                val score = outputBytes[cId]
                val conf = allProbabilities[cId]
                val name = resolveClassName(cId, vocabulary)
                topCandidates.add(
                    PredictionCandidate(
                        rank = rank,
                        classId = cId,
                        className = name,
                        quantizedScore = score,
                        confidence = conf
                    )
                )
            }

            val topCandidate = topCandidates[0]

            return PredictionResult(
                topClassId = topCandidate.classId,
                topClassName = topCandidate.className,
                topConfidence = topCandidate.confidence,
                topQuantizedScore = topCandidate.quantizedScore,
                top5 = topCandidates,
                allProbabilities = allProbabilities
            )
        }

        /**
         * Decode an INT8 output ByteBuffer into a structured PredictionResult.
         * Buffer is rewound before and after extraction.
         */
        fun decode(
            outputBuffer: ByteBuffer,
            vocabulary: List<String>,
            topK: Int = 5,
            scale: Float = OUTPUT_SCALE,
            zeroPoint: Int = OUTPUT_ZERO_POINT
        ): PredictionResult {
            require(outputBuffer.capacity() >= OUTPUT_BUFFER_SIZE_BYTES) {
                "Output buffer capacity ${outputBuffer.capacity()} is less than required $OUTPUT_BUFFER_SIZE_BYTES"
            }
            outputBuffer.rewind()
            val bytes = ByteArray(NUM_CLASSES)
            outputBuffer.get(bytes)
            outputBuffer.rewind()
            return decode(bytes, vocabulary, topK, scale, zeroPoint)
        }
    }
}
