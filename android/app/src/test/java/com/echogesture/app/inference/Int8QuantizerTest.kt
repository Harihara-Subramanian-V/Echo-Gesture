package com.echogesture.app.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Int8QuantizerTest {

    @Test
    fun testZeroMapsToZeroPoint() {
        // Zero in float must map exactly to zero point (23)
        val qZero = Int8Quantizer.quantize(0.0f)
        assertEquals(23.toByte(), qZero)

        // Also test instance method
        val quantizer = Int8Quantizer()
        assertEquals(23.toByte(), quantizer.quantize(0.0f))
    }

    @Test
    fun testExtremeValuesSaturateToBounds() {
        // -100.0f and lower must saturate to -128
        val qNeg100 = Int8Quantizer.quantize(-100.0f)
        assertEquals((-128).toByte(), qNeg100)

        val qNegLarge = Int8Quantizer.quantize(-1000.0f)
        assertEquals((-128).toByte(), qNegLarge)

        // 100.0f and higher must saturate to 127
        val qPos100 = Int8Quantizer.quantize(100.0f)
        assertEquals(127.toByte(), qPos100)

        val qPosLarge = Int8Quantizer.quantize(1000.0f)
        assertEquals(127.toByte(), qPosLarge)
    }

    @Test
    fun testNanMapsToZeroPoint() {
        // Non-finite NaN must be handled safely as 0.0f (mapping to zeroPoint 23)
        val qNan = Int8Quantizer.quantize(Float.NaN)
        assertEquals(23.toByte(), qNan)

        val quantizer = Int8Quantizer()
        assertEquals(23.toByte(), quantizer.quantize(Float.NaN))
    }

    @Test
    fun testInfinitiesSaturateToBounds() {
        val qPosInf = Int8Quantizer.quantize(Float.POSITIVE_INFINITY)
        assertEquals(127.toByte(), qPosInf)

        val qNegInf = Int8Quantizer.quantize(Float.NEGATIVE_INFINITY)
        assertEquals((-128).toByte(), qNegInf)
    }

    @Test
    fun testQuantizationMathKnownValues() {
        // S_in = 0.05794047564268112f, Z_in = 23
        // 1 step above 0: x = S_in -> round(1.0) + 23 = 24
        val qPlusOne = Int8Quantizer.quantize(Int8Quantizer.INPUT_SCALE)
        assertEquals(24.toByte(), qPlusOne)

        // 1 step below 0: x = -S_in -> round(-1.0) + 23 = 22
        val qMinusOne = Int8Quantizer.quantize(-Int8Quantizer.INPUT_SCALE)
        assertEquals(22.toByte(), qMinusOne)

        // 10 steps above 0: 10 * S_in -> round(10.0) + 23 = 33
        val qPlus10 = Int8Quantizer.quantize(10.0f * Int8Quantizer.INPUT_SCALE)
        assertEquals(33.toByte(), qPlus10)
    }

    @Test
    fun testFloatArrayToByteBufferMatchesElementWise() {
        val totalElements = ModelContract.INPUT_BUFFER_SIZE_BYTES // 16,832
        val features = FloatArray(totalElements) { index ->
            // Mix of positive, negative, zero, and fractional values
            ((index % 200) - 100) * 0.05f
        }

        val directBuffer = ByteBuffer.allocateDirect(totalElements).order(ByteOrder.nativeOrder())
        Int8Quantizer.quantize(features, directBuffer)

        // Check buffer was rewound
        assertEquals(0, directBuffer.position())

        // Read bytes from buffer and verify against element-wise quantize
        val extractedBytes = ByteArray(totalElements)
        directBuffer.get(extractedBytes)
        directBuffer.rewind()

        for (i in 0 until totalElements) {
            val expectedByte = Int8Quantizer.quantize(features[i])
            assertEquals("Mismatch at index $i", expectedByte, extractedBytes[i])
        }
    }

    @Test
    fun test2DArrayToByteBufferMatchesElementWise() {
        val seqLen = ModelContract.SEQUENCE_LENGTH // 64
        val featDim = ModelContract.FEATURE_DIM // 263
        val totalElements = seqLen * featDim

        val features2D = Array(seqLen) { t ->
            FloatArray(featDim) { d ->
                (t * 0.1f) - (d * 0.02f)
            }
        }

        val directBuffer = ByteBuffer.allocateDirect(totalElements).order(ByteOrder.nativeOrder())
        Int8Quantizer.quantize(features2D, directBuffer)

        assertEquals(0, directBuffer.position())

        val extractedBytes = ByteArray(totalElements)
        directBuffer.get(extractedBytes)
        directBuffer.rewind()

        var idx = 0
        for (t in 0 until seqLen) {
            for (d in 0 until featDim) {
                val expectedByte = Int8Quantizer.quantize(features2D[t][d])
                assertEquals("Mismatch at frame $t, dim $d", expectedByte, extractedBytes[idx])
                idx++
            }
        }
    }

    @Test
    fun testQuantizeToByteArrayWithAndWithoutDestination() {
        val totalElements = ModelContract.INPUT_BUFFER_SIZE_BYTES
        val features = FloatArray(totalElements) { (it % 50) * 0.1f }

        // Test allocating new ByteArray
        val bytesNew = Int8Quantizer.quantizeToByteArray(features)
        assertEquals(totalElements, bytesNew.size)

        // Test with preallocated destination
        val preallocated = ByteArray(totalElements)
        val returned = Int8Quantizer.quantizeToByteArray(features, preallocated)
        assertTrue(preallocated === returned)

        assertArrayEquals(bytesNew, preallocated)
    }

    @Test
    fun testQuantizeToByteArray2D() {
        val seqLen = ModelContract.SEQUENCE_LENGTH
        val featDim = ModelContract.FEATURE_DIM
        val totalElements = seqLen * featDim

        val features2D = Array(seqLen) { t ->
            FloatArray(featDim) { d -> (t + d) * 0.01f }
        }

        val dest = ByteArray(totalElements)
        Int8Quantizer.quantizeToByteArray(features2D, dest)

        var idx = 0
        for (t in 0 until seqLen) {
            for (d in 0 until featDim) {
                assertEquals(Int8Quantizer.quantize(features2D[t][d]), dest[idx])
                idx++
            }
        }
    }

    @Test
    fun testInvalidInputDimensionsThrow() {
        // Wrong 1D size
        try {
            Int8Quantizer.quantizeToByteArray(FloatArray(100))
            fail("Expected IllegalArgumentException for wrong 1D size")
        } catch (e: IllegalArgumentException) {
            // Expected
        }

        // Wrong 2D sequence length
        try {
            val wrongSeq = Array(63) { FloatArray(ModelContract.FEATURE_DIM) }
            Int8Quantizer.quantizeToByteArray(wrongSeq)
            fail("Expected IllegalArgumentException for wrong sequence length")
        } catch (e: IllegalArgumentException) {
            // Expected
        }

        // Wrong 2D feature dimension
        try {
            val wrongDim = Array(ModelContract.SEQUENCE_LENGTH) { FloatArray(200) }
            Int8Quantizer.quantizeToByteArray(wrongDim)
            fail("Expected IllegalArgumentException for wrong feature dimension")
        } catch (e: IllegalArgumentException) {
            // Expected
        }

        // Destination buffer too small
        try {
            val smallBuffer = ByteBuffer.allocateDirect(100)
            Int8Quantizer.quantize(FloatArray(ModelContract.INPUT_BUFFER_SIZE_BYTES), smallBuffer)
            fail("Expected IllegalArgumentException for small buffer")
        } catch (e: IllegalArgumentException) {
            // Expected
        }
    }
}
