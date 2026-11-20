package com.echogesture.app.inference

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Unit test suite for [EchoGestureInferenceEngine].
 *
 * Verifies end-to-end integration:
 * SequencePreprocessor263D -> Int8Quantizer -> LiteRtModelRunner -> PredictionDecoder.
 */
class EchoGestureInferenceEngineTest {

    private lateinit var engine: EchoGestureInferenceEngine
    private lateinit var casesJson: JSONObject

    @Before
    fun setUp() {
        val modelFile = listOf(
            File("src/main/assets/echogesture_exp_d_int8.tflite"),
            File("app/src/main/assets/echogesture_exp_d_int8.tflite"),
            File("../app/src/main/assets/echogesture_exp_d_int8.tflite")
        ).firstOrNull { it.exists() }
        assertNotNull("Model file must exist", modelFile)

        val vocabFile = listOf(
            File("src/main/assets/label_encoder.json"),
            File("app/src/main/assets/label_encoder.json"),
            File("../app/src/main/assets/label_encoder.json")
        ).firstOrNull { it.exists() }
        assertNotNull("Vocabulary file must exist", vocabFile)

        val scalerFile = listOf(
            File("src/main/assets/scaler_params.json"),
            File("app/src/main/assets/scaler_params.json"),
            File("../app/src/main/assets/scaler_params.json")
        ).firstOrNull { it.exists() }
        assertNotNull("Scaler file must exist", scalerFile)

        val fixtureFile = listOf(
            File("../../tests/fixtures/android_parity/stage3_inference.json"),
            File("../tests/fixtures/android_parity/stage3_inference.json"),
            File("tests/fixtures/android_parity/stage3_inference.json")
        ).firstOrNull { it.exists() }
        assertNotNull("stage3_inference.json fixture file must exist", fixtureFile)

        val fixtureJson = JSONObject(fixtureFile!!.readText(Charsets.UTF_8))
        casesJson = fixtureJson.getJSONObject("cases")

        engine = EchoGestureInferenceEngine.fromFiles(
            modelFile = modelFile!!,
            vocabFile = vocabFile!!,
            scalerFile = scalerFile!!,
            numThreads = 2
        )
    }

    private fun parseFlatFloatArray(jsonArray: org.json.JSONArray): FloatArray {
        val rows = jsonArray.length()
        val cols = jsonArray.getJSONArray(0).length()
        val flat = FloatArray(rows * cols)
        var idx = 0
        for (r in 0 until rows) {
            val row = jsonArray.getJSONArray(r)
            for (c in 0 until cols) {
                flat[idx++] = row.getDouble(c).toFloat()
            }
        }
        return flat
    }

    @Test
    fun testPredictPreprocessed_RealTestSample() {
        val caseObj = casesJson.getJSONObject("case_5_real_test_sample")
        val inputBatch = caseObj.getJSONArray("input_float32")
        val input2D = inputBatch.getJSONArray(0)
        val flat = parseFlatFloatArray(input2D)

        val result = engine.predictPreprocessed(flat)

        // Verify output structure and dimensions
        assertEquals(5, result.top5.size)
        assertEquals(1, result.top5[0].rank)
        assertEquals(result.topClassId, result.top5[0].classId)
        assertEquals(result.topClassName, result.top5[0].className)
        assertEquals(40, result.allProbabilities.size)

        // Verify Top-5 monotonic sorting
        for (i in 0 until result.top5.size - 1) {
            assertTrue("Top-5 confidences must be descending", result.top5[i].confidence >= result.top5[i + 1].confidence)
        }
    }

    @Test
    fun testPredictPreprocessed_AllZeros() {
        val caseObj = casesJson.getJSONObject("case_1_canonical_all_zeros")
        val inputBatch = caseObj.getJSONArray("input_float32")
        val input2D = inputBatch.getJSONArray(0)
        val flat = parseFlatFloatArray(input2D)

        val result = engine.predictPreprocessed(flat)

        assertEquals(5, result.top5.size)
        assertEquals(1, result.top5[0].rank)
        assertEquals(result.topClassId, result.top5[0].classId)
        assertTrue(result.topConfidence in 0.0f..1.0f)
    }

    @Test
    fun testPredictQuantizedDirectly() {
        val testBytes = ByteArray(16832) { 23 } // Affine zero-point
        val result = engine.predictQuantized(testBytes)

        assertEquals(5, result.top5.size)
        assertEquals(40, result.allProbabilities.size)
        assertTrue(result.topConfidence in 0.0f..1.0f)
    }
}
