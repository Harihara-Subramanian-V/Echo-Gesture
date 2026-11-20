package com.echogesture.app.inference

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Production contract and quantization specification for EchoGesture INT8 model.
 * Invariants strictly match reports/production_inference_contract.md and reports/phase6f_mobile_compatibility_audit.md.
 */
object ModelContract {
    const val MODEL_ASSET_PATH = "echogesture_exp_d_int8.tflite"
    const val LABEL_ENCODER_PATH = "label_encoder.json"
    const val EXPECTED_SHA256 = "ABD726BC87F9E367653A007CBDC2B3AA4AE6371AF54883523A3CAC6811273499"

    const val BATCH_SIZE = 1
    const val SEQUENCE_LENGTH = 64
    const val FEATURE_DIM = 263
    const val NUM_CLASSES = 40

    const val INPUT_BUFFER_SIZE_BYTES = BATCH_SIZE * SEQUENCE_LENGTH * FEATURE_DIM // 16,832 bytes
    const val OUTPUT_BUFFER_SIZE_BYTES = BATCH_SIZE * NUM_CLASSES // 40 bytes

    // Affine Quantization Parameters for Input Tensor
    const val INPUT_SCALE = 0.05794047564268112f
    const val INPUT_ZERO_POINT = 23

    // Affine Quantization Parameters for Output Tensor
    const val OUTPUT_SCALE = 0.00390625f // 1.0f / 256.0f
    const val OUTPUT_ZERO_POINT = -128

    /**
     * Quantize a single float32 feature value to signed int8 byte in [-128, 127].
     * Math: q = clamp(round(x / S_in) + Z_in, -128, 127)
     */
    fun quantize(floatValue: Float): Byte {
        val q = (floatValue / INPUT_SCALE).roundToInt() + INPUT_ZERO_POINT
        return q.coerceIn(-128, 127).toByte()
    }

    /**
     * Dequantize a signed int8 byte to a float32 probability in [0.0, 1.0].
     * Math: p = (q - Z_out) * S_out = (q + 128) / 256.0
     */
    fun dequantize(byteValue: Byte): Float {
        val qInt = byteValue.toInt() // in [-128, 127]
        return (qInt - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
    }

    /**
     * Allocate a DirectByteBuffer with native byte order for zero-overhead JNI execution.
     */
    fun createDirectByteBuffer(capacityBytes: Int): ByteBuffer {
        return ByteBuffer.allocateDirect(capacityBytes).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Finds the index of the highest value in a byte array (argmax).
     * Because dequantization is strictly monotonic, argmax(byte) == argmax(float).
     */
    fun computeArgMax(outputBytes: ByteArray): Int {
        if (outputBytes.isEmpty()) return -1
        var maxIndex = 0
        var maxValue = outputBytes[0]
        for (i in 1 until outputBytes.size) {
            if (outputBytes[i] > maxValue) {
                maxValue = outputBytes[i]
                maxIndex = i
            }
        }
        return maxIndex
    }
}
