package com.echogesture.app.inference

import java.nio.ByteBuffer
import kotlin.math.round

/**
 * High-performance INT8 affine quantizer for EchoGesture mobile inference.
 *
 * Implements mathematical quantization aligned with:
 * - reports/production_inference_contract.md (Stage 7)
 * - scripts/echogesture_runtime.py (Lines 105-109)
 * - ModelContract specification
 *
 * Affine Quantization Formula:
 *     q = clip(round(x / S_in) + Z_in, -128, 127)
 *
 * Scale S_in = 0.05794047564268112f
 * Zero Point Z_in = 23
 */
class Int8Quantizer(
    val scale: Float = ModelContract.INPUT_SCALE,
    val zeroPoint: Int = ModelContract.INPUT_ZERO_POINT
) {
    /**
     * Quantize a single float32 feature value to signed int8 byte in [-128, 127].
     */
    fun quantize(floatValue: Float): Byte = Companion.quantize(floatValue, scale, zeroPoint)

    /**
     * Quantize a flat FloatArray of size 16,832 directly into a destination ByteBuffer.
     */
    fun quantize(features: FloatArray, destination: ByteBuffer) =
        Companion.quantize(features, destination, scale, zeroPoint)

    /**
     * Quantize a 2D Array<FloatArray> of shape [64, 263] directly into a destination ByteBuffer.
     */
    fun quantize(features: Array<FloatArray>, destination: ByteBuffer) =
        Companion.quantize(features, destination, scale, zeroPoint)

    /**
     * Quantize a flat FloatArray of size 16,832 to a ByteArray.
     */
    fun quantizeToByteArray(features: FloatArray, destination: ByteArray? = null): ByteArray =
        Companion.quantizeToByteArray(features, destination, scale, zeroPoint)

    /**
     * Quantize a 2D Array<FloatArray> of shape [64, 263] to a flat ByteArray of size 16,832.
     */
    fun quantizeToByteArray(features: Array<FloatArray>, destination: ByteArray? = null): ByteArray =
        Companion.quantizeToByteArray(features, destination, scale, zeroPoint)

    companion object {
        const val INPUT_SCALE: Float = ModelContract.INPUT_SCALE
        const val INPUT_ZERO_POINT: Int = ModelContract.INPUT_ZERO_POINT
        const val INPUT_BUFFER_SIZE_BYTES: Int = ModelContract.INPUT_BUFFER_SIZE_BYTES
        const val SEQUENCE_LENGTH: Int = ModelContract.SEQUENCE_LENGTH
        const val FEATURE_DIM: Int = ModelContract.FEATURE_DIM

        /**
         * Quantize a single float32 feature value to signed int8 byte in [-128, 127].
         *
         * Edge-case handling:
         * - NaN maps to 0.0f (quantized value Z_in = 23).
         * - Positive extreme / +Infinity saturates to 127.
         * - Negative extreme / -Infinity saturates to -128.
         */
        fun quantize(
            floatValue: Float,
            scale: Float = INPUT_SCALE,
            zeroPoint: Int = INPUT_ZERO_POINT
        ): Byte {
            if (floatValue.isNaN()) {
                return zeroPoint.toByte()
            }
            val scaled = floatValue / scale
            if (scaled >= 256.0f) {
                return 127.toByte()
            }
            if (scaled <= -256.0f) {
                return (-128).toByte()
            }
            val q = round(scaled).toInt() + zeroPoint
            return q.coerceIn(-128, 127).toByte()
        }

        /**
         * Quantize a flat FloatArray of size 16,832 directly into a destination ByteBuffer.
         * Buffer is rewound before and after population for zero-copy LiteRT execution.
         */
        fun quantize(
            features: FloatArray,
            destination: ByteBuffer,
            scale: Float = INPUT_SCALE,
            zeroPoint: Int = INPUT_ZERO_POINT
        ) {
            require(features.size == INPUT_BUFFER_SIZE_BYTES) {
                "Expected features size $INPUT_BUFFER_SIZE_BYTES, got ${features.size}"
            }
            require(destination.capacity() >= INPUT_BUFFER_SIZE_BYTES) {
                "Destination buffer capacity ${destination.capacity()} is less than required $INPUT_BUFFER_SIZE_BYTES"
            }
            require(!destination.isReadOnly) {
                "Destination buffer must not be read-only"
            }

            destination.rewind()
            for (i in 0 until INPUT_BUFFER_SIZE_BYTES) {
                destination.put(quantize(features[i], scale, zeroPoint))
            }
            destination.rewind()
        }

        /**
         * Quantize a 2D Array<FloatArray> of shape [64, 263] directly into a destination ByteBuffer.
         * Buffer is rewound before and after population for zero-copy LiteRT execution.
         */
        fun quantize(
            features: Array<FloatArray>,
            destination: ByteBuffer,
            scale: Float = INPUT_SCALE,
            zeroPoint: Int = INPUT_ZERO_POINT
        ) {
            require(features.size == SEQUENCE_LENGTH) {
                "Expected sequence length $SEQUENCE_LENGTH, got ${features.size}"
            }
            require(destination.capacity() >= INPUT_BUFFER_SIZE_BYTES) {
                "Destination buffer capacity ${destination.capacity()} is less than required $INPUT_BUFFER_SIZE_BYTES"
            }
            require(!destination.isReadOnly) {
                "Destination buffer must not be read-only"
            }

            destination.rewind()
            for (t in 0 until SEQUENCE_LENGTH) {
                val frame = features[t]
                require(frame.size == FEATURE_DIM) {
                    "Expected frame $t to have dimension $FEATURE_DIM, got ${frame.size}"
                }
                for (d in 0 until FEATURE_DIM) {
                    destination.put(quantize(frame[d], scale, zeroPoint))
                }
            }
            destination.rewind()
        }

        /**
         * Quantize a flat FloatArray of size 16,832 into a flat ByteArray.
         */
        fun quantizeToByteArray(
            features: FloatArray,
            destination: ByteArray? = null,
            scale: Float = INPUT_SCALE,
            zeroPoint: Int = INPUT_ZERO_POINT
        ): ByteArray {
            require(features.size == INPUT_BUFFER_SIZE_BYTES) {
                "Expected features size $INPUT_BUFFER_SIZE_BYTES, got ${features.size}"
            }
            val out = destination ?: ByteArray(INPUT_BUFFER_SIZE_BYTES)
            require(out.size == INPUT_BUFFER_SIZE_BYTES) {
                "Destination size ${out.size} does not match required $INPUT_BUFFER_SIZE_BYTES"
            }

            for (i in 0 until INPUT_BUFFER_SIZE_BYTES) {
                out[i] = quantize(features[i], scale, zeroPoint)
            }
            return out
        }

        /**
         * Quantize a 2D Array<FloatArray> of shape [64, 263] into a flat ByteArray of size 16,832.
         */
        fun quantizeToByteArray(
            features: Array<FloatArray>,
            destination: ByteArray? = null,
            scale: Float = INPUT_SCALE,
            zeroPoint: Int = INPUT_ZERO_POINT
        ): ByteArray {
            require(features.size == SEQUENCE_LENGTH) {
                "Expected sequence length $SEQUENCE_LENGTH, got ${features.size}"
            }
            val out = destination ?: ByteArray(INPUT_BUFFER_SIZE_BYTES)
            require(out.size == INPUT_BUFFER_SIZE_BYTES) {
                "Destination size ${out.size} does not match required $INPUT_BUFFER_SIZE_BYTES"
            }

            var offset = 0
            for (t in 0 until SEQUENCE_LENGTH) {
                val frame = features[t]
                require(frame.size == FEATURE_DIM) {
                    "Expected frame $t to have dimension $FEATURE_DIM, got ${frame.size}"
                }
                for (d in 0 until FEATURE_DIM) {
                    out[offset++] = quantize(frame[d], scale, zeroPoint)
                }
            }
            return out
        }
    }
}
