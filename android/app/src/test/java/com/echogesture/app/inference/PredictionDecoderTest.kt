package com.echogesture.app.inference

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PredictionDecoderTest {

    private val sampleVocabulary = listOf(
        "all", "before", "cool", "fine", "give",
        "hello", "kiss", "later", "nice", "owl",
        "pizza", "quiet", "red", "sorry", "thankyou",
        "ugly", "visit", "water", "yellow", "zero",
        "accept", "balance", "catch", "dance", "eagle",
        "family", "game", "happy", "island", "jacket",
        "kitchen", "learn", "music", "night", "orange",
        "pencil", "queen", "river", "school", "yes"
    )

    @Test
    fun testDequantizeMinByteMapsToZero() {
        // Byte -128 dequantizes to 0.0f
        val pZero = PredictionDecoder.dequantize((-128).toByte())
        assertEquals(0.0f, pZero, 1e-7f)

        // Also test instance method
        val decoder = PredictionDecoder()
        assertEquals(0.0f, decoder.dequantize((-128).toByte()), 1e-7f)
    }

    @Test
    fun testDequantizeMaxByteMapsTo255Over256() {
        // Byte 127 dequantizes to 255 / 256 (~0.99609375f)
        val pMax = PredictionDecoder.dequantize(127.toByte())
        val expected = 255.0f / 256.0f
        assertEquals(expected, pMax, 1e-7f)

        val decoder = PredictionDecoder()
        assertEquals(expected, decoder.dequantize(127.toByte()), 1e-7f)
    }

    @Test
    fun testDequantizeZeroByteMapsToHalf() {
        // Byte 0 dequantizes to (0 - (-128)) / 256 = 128 / 256 = 0.5f
        val pHalf = PredictionDecoder.dequantize(0.toByte())
        assertEquals(0.5f, pHalf, 1e-7f)
    }

    @Test
    fun testDequantizeAllParityAndDestinationReuse() {
        val bytes = ByteArray(ModelContract.NUM_CLASSES) { i ->
            (i * 6 - 120).toByte() // Diverse values
        }

        val probsNew = PredictionDecoder.dequantizeAll(bytes)
        assertEquals(ModelContract.NUM_CLASSES, probsNew.size)

        for (i in 0 until ModelContract.NUM_CLASSES) {
            val expected = PredictionDecoder.dequantize(bytes[i])
            assertEquals(expected, probsNew[i], 1e-7f)
        }

        // Test with preallocated destination
        val preallocated = FloatArray(ModelContract.NUM_CLASSES)
        val returned = PredictionDecoder.dequantizeAll(bytes, preallocated)
        assertTrue(preallocated === returned)
        assertArrayEquals(probsNew, preallocated, 1e-7f)
    }

    @Test
    fun testTop5SortingIsStrictlyDescending() {
        val bytes = ByteArray(ModelContract.NUM_CLASSES) { (-100).toByte() }
        // Set distinct scores for known classes
        bytes[10] = 50.toByte()   // Rank 3
        bytes[23] = 95.toByte()   // Rank 1 (Top class)
        bytes[4] = 80.toByte()    // Rank 2
        bytes[39] = 30.toByte()   // Rank 4
        bytes[0] = 10.toByte()    // Rank 5
        bytes[15] = (-20).toByte()  // Rank 6

        val result = PredictionDecoder.decode(bytes, sampleVocabulary)

        // Verify top-1 prediction
        assertEquals(23, result.topClassId)
        assertEquals("dance", result.topClassName)
        assertEquals(95.toByte(), result.topQuantizedScore)
        assertEquals(PredictionDecoder.dequantize(95.toByte()), result.topConfidence, 1e-7f)

        // Verify top-5 candidates list
        assertEquals(5, result.top5.size)

        val expectedRanks = listOf(1, 2, 3, 4, 5)
        val expectedClassIds = listOf(23, 4, 10, 39, 0)
        val expectedClassNames = listOf("dance", "give", "pizza", "yes", "all")
        val expectedScores = listOf(95.toByte(), 80.toByte(), 50.toByte(), 30.toByte(), 10.toByte())

        for (i in 0 until 5) {
            val candidate = result.top5[i]
            assertEquals(expectedRanks[i], candidate.rank)
            assertEquals(expectedClassIds[i], candidate.classId)
            assertEquals(expectedClassNames[i], candidate.className)
            assertEquals(expectedScores[i], candidate.quantizedScore)
            assertEquals(PredictionDecoder.dequantize(expectedScores[i]), candidate.confidence, 1e-7f)
        }

        // Verify strictly descending confidences
        for (i in 0 until result.top5.size - 1) {
            assertTrue(
                "Expected top5[${i}].confidence >= top5[${i + 1}].confidence",
                result.top5[i].confidence >= result.top5[i + 1].confidence
            )
            assertTrue(
                "Expected top5[${i}].quantizedScore >= top5[${i + 1}].quantizedScore",
                result.top5[i].quantizedScore >= result.top5[i + 1].quantizedScore
            )
        }
    }

    @Test
    fun testVocabularyMappingMatchesIndices() {
        assertEquals(40, sampleVocabulary.size)
        val bytes = ByteArray(ModelContract.NUM_CLASSES) { (-120).toByte() }

        for (idx in 0 until ModelContract.NUM_CLASSES) {
            bytes.fill((-120).toByte())
            bytes[idx] = 100.toByte()

            val result = PredictionDecoder.decode(bytes, sampleVocabulary)
            assertEquals(idx, result.topClassId)
            assertEquals(sampleVocabulary[idx], result.topClassName)
            assertEquals(sampleVocabulary[idx], result.top5[0].className)
        }
    }

    @Test
    fun testFallbackToClassIdWhenVocabularyIncompleteOrEmpty() {
        val bytes = ByteArray(ModelContract.NUM_CLASSES) { (-100).toByte() }
        bytes[7] = 88.toByte()
        bytes[25] = 44.toByte()

        // 1. Completely empty vocabulary
        val emptyResult = PredictionDecoder.decode(bytes, emptyList())
        assertEquals(7, emptyResult.topClassId)
        assertEquals("class_7", emptyResult.topClassName)
        assertEquals("class_7", emptyResult.top5[0].className)
        assertEquals("class_25", emptyResult.top5[1].className)

        // 2. Truncated vocabulary (only 5 items)
        val shortVocab = listOf("zero", "one", "two", "three", "four")
        val shortResult = PredictionDecoder.decode(bytes, shortVocab)
        assertEquals(7, shortResult.topClassId)
        assertEquals("class_7", shortResult.topClassName)
    }

    @Test
    fun testByteBufferDecodingMatchesByteArray() {
        val bytes = ByteArray(ModelContract.NUM_CLASSES) { i ->
            ((i * 13) % 256 - 128).toByte()
        }

        val directBuffer = ByteBuffer.allocateDirect(ModelContract.OUTPUT_BUFFER_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
        directBuffer.put(bytes)
        directBuffer.rewind()

        val fromBytes = PredictionDecoder.decode(bytes, sampleVocabulary)
        val fromBuffer = PredictionDecoder.decode(directBuffer, sampleVocabulary)

        assertEquals(0, directBuffer.position())
        assertEquals(fromBytes.topClassId, fromBuffer.topClassId)
        assertEquals(fromBytes.topClassName, fromBuffer.topClassName)
        assertEquals(fromBytes.topConfidence, fromBuffer.topConfidence, 1e-7f)
        assertEquals(fromBytes.topQuantizedScore, fromBuffer.topQuantizedScore)
        assertEquals(fromBytes.top5, fromBuffer.top5)
        assertArrayEquals(fromBytes.allProbabilities, fromBuffer.allProbabilities, 1e-7f)
    }

    @Test
    fun testInvalidOutputDimensionsThrow() {
        // Wrong byte array size
        try {
            PredictionDecoder.decode(ByteArray(39), sampleVocabulary)
            fail("Expected IllegalArgumentException for 39 bytes")
        } catch (e: IllegalArgumentException) {
            // Expected
        }

        try {
            PredictionDecoder.decode(ByteArray(41), sampleVocabulary)
            fail("Expected IllegalArgumentException for 41 bytes")
        } catch (e: IllegalArgumentException) {
            // Expected
        }

        // Buffer too small
        try {
            val smallBuffer = ByteBuffer.allocateDirect(20)
            PredictionDecoder.decode(smallBuffer, sampleVocabulary)
            fail("Expected IllegalArgumentException for buffer capacity 20")
        } catch (e: IllegalArgumentException) {
            // Expected
        }
    }
}
