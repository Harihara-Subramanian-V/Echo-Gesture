package com.echogesture.app

import com.echogesture.app.inference.ModelContract
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest

class ModelContractTest {

    @Test
    fun testTensorDimensionsAndConstants() {
        assertEquals(1, ModelContract.BATCH_SIZE)
        assertEquals(64, ModelContract.SEQUENCE_LENGTH)
        assertEquals(263, ModelContract.FEATURE_DIM)
        assertEquals(40, ModelContract.NUM_CLASSES)
        assertEquals(16832, ModelContract.INPUT_BUFFER_SIZE_BYTES)
        assertEquals(40, ModelContract.OUTPUT_BUFFER_SIZE_BYTES)
    }

    @Test
    fun testQuantizationMath() {
        // Zero in float maps to zero point (23)
        val qZero = ModelContract.quantize(0.0f)
        assertEquals(23.toByte(), qZero)

        // Extreme bounds clamp to -128 and 127
        val qNeg = ModelContract.quantize(-100.0f)
        assertEquals((-128).toByte(), qNeg)

        val qPos = ModelContract.quantize(100.0f)
        assertEquals(127.toByte(), qPos)
    }

    @Test
    fun testDequantizationMath() {
        // Zero probability corresponds to byte -128
        val pZero = ModelContract.dequantize((-128).toByte())
        assertEquals(0.0f, pZero, 1e-6f)

        // Max logit byte 127 corresponds to ~0.996
        val pMax = ModelContract.dequantize(127.toByte())
        assertEquals(255f / 256f, pMax, 1e-6f)
    }

    @Test
    fun testArgMaxMonotonicity() {
        val testBytes = byteArrayOf(-120, -50, 42, 10, -100)
        val maxIdx = ModelContract.computeArgMax(testBytes)
        assertEquals(2, maxIdx)
        assertEquals(42.toByte(), testBytes[maxIdx])
    }

    @Test
    fun testModelAssetSha256Integrity() {
        // Locate model in assets directory relative to project root
        val possiblePaths = listOf(
            File("src/main/assets/echogesture_exp_d_int8.tflite"),
            File("app/src/main/assets/echogesture_exp_d_int8.tflite"),
            File("../app/src/main/assets/echogesture_exp_d_int8.tflite")
        )
        val modelFile = possiblePaths.firstOrNull { it.exists() }
        assertTrue("Model file should exist in assets", modelFile != null && modelFile.exists())

        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = modelFile!!.readBytes()
        val hashBytes = digest.digest(bytes)
        val computedHash = hashBytes.joinToString("") { "%02X".format(it) }

        assertEquals(ModelContract.EXPECTED_SHA256, computedHash)
        assertEquals(135464L, modelFile.length())
    }

    @Test
    fun testVocabularyAssetIntegrity() {
        val possiblePaths = listOf(
            File("src/main/assets/label_encoder.json"),
            File("app/src/main/assets/label_encoder.json"),
            File("../app/src/main/assets/label_encoder.json")
        )
        val jsonFile = possiblePaths.firstOrNull { it.exists() }
        assertTrue("label_encoder.json should exist in assets", jsonFile != null && jsonFile.exists())

        val json = JSONObject(jsonFile!!.readText())
        val idx2label = json.getJSONObject("idx2label")
        assertEquals(40, idx2label.length())
        assertEquals("all", idx2label.getString("0"))
        assertEquals("yes", idx2label.getString("39"))
    }
}
