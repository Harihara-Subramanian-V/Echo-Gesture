package com.echogesture.app.preprocessing

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Unit test suite for [SequencePreprocessor263D].
 *
 * Validates the fused end-to-end preprocessing pipeline entry point against
 * the Stage 2 reference fixture `stage2_temporal.json`.
 */
class SequencePreprocessor263DTest {

    private lateinit var preprocessor: SequencePreprocessor263D
    private lateinit var rootJson: JSONObject
    private lateinit var casesJson: JSONObject

    @Before
    fun setUp() {
        val fixtureFile = listOf(
            File("../../tests/fixtures/android_parity/stage2_temporal.json"),
            File("../tests/fixtures/android_parity/stage2_temporal.json"),
            File("tests/fixtures/android_parity/stage2_temporal.json")
        ).firstOrNull { it.exists() }
        assertNotNull("stage2_temporal.json fixture file must exist", fixtureFile)

        val jsonStr = fixtureFile!!.readText(Charsets.UTF_8)
        rootJson = JSONObject(jsonStr)
        casesJson = rootJson.getJSONObject("cases")

        preprocessor = SequencePreprocessor263D.fromJson(jsonStr)
    }

    private fun parse2DArray(jsonArray: org.json.JSONArray): Array<FloatArray> {
        val rows = jsonArray.length()
        val result = Array(rows) { FloatArray(0) }
        for (r in 0 until rows) {
            val rowArr = jsonArray.getJSONArray(r)
            val cols = rowArr.length()
            val row = FloatArray(cols)
            for (c in 0 until cols) {
                row[c] = rowArr.getDouble(c).toFloat()
            }
            result[r] = row
        }
        return result
    }

    @Test
    fun testProcessCase1_ShortSequence() {
        val caseJson = casesJson.getJSONObject("case_1_short_sequence")
        val rawInput = parse2DArray(caseJson.getJSONArray("raw_input_134"))
        val expectedOut = parse2DArray(caseJson.getJSONArray("output_263"))

        val actualOut = preprocessor.process(rawInput)

        assertEquals("Output row count must be 64", 64, actualOut.size)
        assertEquals("Output feature dimension must be 263", 263, actualOut[0].size)

        var maxErr = 0.0f
        for (t in 0 until 64) {
            for (c in 0 until 263) {
                val err = abs(actualOut[t][c] - expectedOut[t][c])
                if (err > maxErr) maxErr = err
                assertTrue("Case 1 mismatch at [$t, $c]: actual=${actualOut[t][c]}, expected=${expectedOut[t][c]}", err <= 2.0e-5f)
            }
        }
        assertTrue("Max error must be <= 2.0e-5, was $maxErr", maxErr <= 2.0e-5f)
    }

    @Test
    fun testProcessCase8_IdleWindow() {
        val caseJson = casesJson.getJSONObject("case_8_idle_window")
        val rawInput = parse2DArray(caseJson.getJSONArray("raw_input_134"))

        val actualOut = preprocessor.process(rawInput)

        for (t in 0 until 64) {
            for (c in 0 until 263) {
                assertEquals("Idle window output must be strictly 0.0 at [$t, $c]", 0.0f, actualOut[t][c], 0.0f)
            }
        }
    }

    @Test
    fun testProcessToFlat_matches2DOutput() {
        val caseJson = casesJson.getJSONObject("case_2_full_sequence")
        val rawInput = parse2DArray(caseJson.getJSONArray("raw_input_134"))

        val actual2D = preprocessor.process(rawInput)
        val actualFlat = preprocessor.processToFlat(rawInput)

        assertEquals("Flat array length must be 64 * 263 = 16832", 64 * 263, actualFlat.size)

        var idx = 0
        for (t in 0 until 64) {
            for (c in 0 until 263) {
                assertEquals("Flat element at index $idx must match 2D [$t, $c]", actual2D[t][c], actualFlat[idx], 0.0f)
                idx++
            }
        }
    }

    @Test
    fun testDestinationBufferReuse() {
        val caseJson = casesJson.getJSONObject("case_6_two_hand_continuous")
        val rawInput = parse2DArray(caseJson.getJSONArray("raw_input_134"))

        val dirtyDest = Array(64) { FloatArray(263) { 999.0f } }
        val out = preprocessor.process(rawInput, dirtyDest)

        assertTrue("Destination array should be returned", out === dirtyDest)
        // Verify not dirty
        for (t in 0 until 64) {
            assertTrue("Values should be reasonable, not dirty 999.0f", out[t][0] < 50.0f)
        }
    }
}
