package com.echogesture.app.preprocessing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Stage 2 Temporal Resampling Parity Validation Test Suite for EchoGesture Phase 7B.3.
 *
 * Validates the Kotlin [TemporalResampler] implementation against:
 * 1. The canonical Stage 2 Python reference fixtures in `tests/fixtures/android_parity/stage2_temporal.json`.
 * 2. Strict zero-masking on inactive frames.
 * 3. Nearest-neighbor presence sampling matching IEEE-754 round-to-nearest-even.
 * 4. Boundary cases: T = 0 (empty), T = 1 (single frame), sparse detections (1 active frame).
 * 5. Destination buffer reuse for zero-allocation real-time Android execution.
 */
class TemporalResamplerTest {

    private lateinit var rootJson: JSONObject
    private lateinit var casesJson: JSONObject
    private lateinit var resampler: TemporalResampler

    companion object {
        private const val PARITY_TOLERANCE = 1e-5f
        private const val STRICT_ZERO = 0.0f
    }

    data class ParityStats(
        val caseName: String,
        val maxAbsError: Float,
        val meanAbsError: Float,
        val mismatchCount: Int,
        val totalElements: Int
    )

    @Before
    fun setUp() {
        rootJson = loadJson()
        casesJson = rootJson.getJSONObject("cases")
        resampler = TemporalResampler()
    }

    private fun loadJson(): JSONObject {
        val possiblePaths = listOf(
            File("src/test/resources/stage2_temporal.json"),
            File("../../tests/fixtures/android_parity/stage2_temporal.json"),
            File("../tests/fixtures/android_parity/stage2_temporal.json"),
            File("tests/fixtures/android_parity/stage2_temporal.json")
        )
        val jsonFile = possiblePaths.firstOrNull { it.exists() }
        assertNotNull(
            "stage2_temporal.json fixture file must exist in one of: ${possiblePaths.map { it.absolutePath }}",
            jsonFile
        )
        return JSONObject(jsonFile!!.readText(Charsets.UTF_8))
    }

    private fun parseMatrix(jsonArray: JSONArray, expectedCols: Int = 134): Array<FloatArray> {
        val rows = jsonArray.length()
        return Array(rows) { r ->
            val rowArray = jsonArray.getJSONArray(r)
            assertEquals("Row $r must have length $expectedCols", expectedCols, rowArray.length())
            FloatArray(expectedCols) { c ->
                rowArray.getDouble(c).toFloat()
            }
        }
    }

    private fun verifyParity(
        caseName: String,
        actual: Array<FloatArray>,
        expected: Array<FloatArray>,
        tolerance: Float = PARITY_TOLERANCE
    ): ParityStats {
        val targetLen = expected.size
        assertEquals("$caseName: actual frame count must be $targetLen", targetLen, actual.size)

        var maxErr = 0.0f
        var sumErr = 0.0
        var mismatches = 0
        val totalElements = targetLen * TemporalResampler.FEATURE_DIM

        for (j in 0 until targetLen) {
            val actualRow = actual[j]
            val expectedRow = expected[j]
            assertEquals("$caseName: frame $j feature dim must be 134", TemporalResampler.FEATURE_DIM, actualRow.size)
            assertEquals("$caseName: frame $j expected dim must be 134", TemporalResampler.FEATURE_DIM, expectedRow.size)

            // 1. Exact equality for binary flags
            assertEquals(
                "$caseName [frame $j]: Left hand presence flag (dim 126) must match exactly",
                expectedRow[TemporalResampler.INDEX_PRES_LEFT],
                actualRow[TemporalResampler.INDEX_PRES_LEFT],
                STRICT_ZERO
            )
            assertEquals(
                "$caseName [frame $j]: Right hand presence flag (dim 127) must match exactly",
                expectedRow[TemporalResampler.INDEX_PRES_RIGHT],
                actualRow[TemporalResampler.INDEX_PRES_RIGHT],
                STRICT_ZERO
            )
            assertEquals(
                "$caseName [frame $j]: Both hands presence flag (dim 130) must match exactly",
                expectedRow[TemporalResampler.INDEX_BOTH_PRESENT],
                actualRow[TemporalResampler.INDEX_BOTH_PRESENT],
                STRICT_ZERO
            )

            // 2. Strict zero-masking checks
            if (expectedRow[TemporalResampler.INDEX_PRES_LEFT] <= 0.5f) {
                for (c in 0 until 63) {
                    assertEquals(
                        "$caseName [frame $j]: Inactive left hand coord $c must be strictly 0.0f",
                        0.0f,
                        actualRow[c],
                        STRICT_ZERO
                    )
                }
                assertEquals(
                    "$caseName [frame $j]: Inactive left hand score must be strictly 0.0f",
                    0.0f,
                    actualRow[TemporalResampler.INDEX_SCORE_LEFT],
                    STRICT_ZERO
                )
            }

            if (expectedRow[TemporalResampler.INDEX_PRES_RIGHT] <= 0.5f) {
                for (c in 0 until 63) {
                    val idx = TemporalResampler.SLOT1_START + c
                    assertEquals(
                        "$caseName [frame $j]: Inactive right hand coord $c (idx $idx) must be strictly 0.0f",
                        0.0f,
                        actualRow[idx],
                        STRICT_ZERO
                    )
                }
                assertEquals(
                    "$caseName [frame $j]: Inactive right hand score must be strictly 0.0f",
                    0.0f,
                    actualRow[TemporalResampler.INDEX_SCORE_RIGHT],
                    STRICT_ZERO
                )
            }

            if (expectedRow[TemporalResampler.INDEX_BOTH_PRESENT] <= 0.5f) {
                assertEquals(
                    "$caseName [frame $j]: Inactive distance (dim 131) must be strictly 0.0f",
                    0.0f,
                    actualRow[TemporalResampler.INDEX_INTER_DIST],
                    STRICT_ZERO
                )
                assertEquals(
                    "$caseName [frame $j]: Inactive offset dx (dim 132) must be strictly 0.0f",
                    0.0f,
                    actualRow[TemporalResampler.INDEX_INTER_OFFSET_DX],
                    STRICT_ZERO
                )
                assertEquals(
                    "$caseName [frame $j]: Inactive offset dy (dim 133) must be strictly 0.0f",
                    0.0f,
                    actualRow[TemporalResampler.INDEX_INTER_OFFSET_DY],
                    STRICT_ZERO
                )
            }

            // 3. Numerical diff across all 134 dimensions
            for (dim in 0 until TemporalResampler.FEATURE_DIM) {
                val actVal = actualRow[dim]
                val expVal = expectedRow[dim]
                val err = abs(actVal - expVal)
                if (err > maxErr) maxErr = err
                sumErr += err.toDouble()
                if (err > tolerance) {
                    mismatches++
                }
            }
        }

        val meanErr = (sumErr / totalElements).toFloat()
        val stats = ParityStats(caseName, maxErr, meanErr, mismatches, totalElements)
        println("Parity [$caseName]: maxErr=$maxErr, meanErr=$meanErr, mismatches=$mismatches / $totalElements")
        assertEquals("$caseName had $mismatches values exceeding tolerance $tolerance (maxErr=$maxErr)", 0, mismatches)
        return stats
    }

    private fun runCaseParity(caseKey: String) {
        assertTrue("Case $caseKey must exist in fixture", casesJson.has(caseKey))
        val caseObj = casesJson.getJSONObject(caseKey)
        val rawInput = parseMatrix(caseObj.getJSONArray("raw_input_134"))
        val expectedResampled = parseMatrix(caseObj.getJSONArray("resampled_64"))

        val actual = resampler.resample(rawInput, 64)
        verifyParity(caseKey, actual, expectedResampled)
    }

    @Test
    fun testCase1_shortSequence_T24() {
        runCaseParity("case_1_short_sequence")
    }

    @Test
    fun testCase2_fullSequence_T64() {
        runCaseParity("case_2_full_sequence")
    }

    @Test
    fun testCase3_longSequence_T96() {
        runCaseParity("case_3_long_sequence")
    }

    @Test
    fun testCase4_sparseDetections_T50() {
        runCaseParity("case_4_sparse_detections")
    }

    @Test
    fun testCase5_singleHandContinuous() {
        runCaseParity("case_5_single_hand_continuous")
    }

    @Test
    fun testCase6_twoHandContinuous() {
        runCaseParity("case_6_two_hand_continuous")
    }

    @Test
    fun testCase7_alternatingHands() {
        runCaseParity("case_7_alternating_hands")
    }

    @Test
    fun testCase8_idleWindow() {
        runCaseParity("case_8_idle_window")
    }

    @Test
    fun testAllStage2CasesParity() {
        val keys = casesJson.keys()
        var caseCount = 0
        while (keys.hasNext()) {
            val key = keys.next()
            runCaseParity(key)
            caseCount++
        }
        assertEquals("Must run all 8 Stage 2 test fixture cases", 8, caseCount)
    }

    @Test
    fun testEdgeCase_emptySequence_T0() {
        val emptyInput = emptyArray<FloatArray>()
        val result = resampler.resample(emptyInput, 64)
        assertEquals(64, result.size)
        for (j in 0 until 64) {
            assertEquals(134, result[j].size)
            for (dim in 0 until 134) {
                assertEquals(0.0f, result[j][dim], STRICT_ZERO)
            }
        }
    }

    @Test
    fun testEdgeCase_singleFrameSequence_T1_active() {
        val singleFrame = Array(1) { FloatArray(134) }
        // Left hand active
        for (c in 0 until 63) {
            singleFrame[0][c] = (c + 1) * 0.01f
        }
        singleFrame[0][TemporalResampler.INDEX_PRES_LEFT] = 1.0f
        singleFrame[0][TemporalResampler.INDEX_SCORE_LEFT] = 0.95f

        // Right hand inactive
        singleFrame[0][TemporalResampler.INDEX_PRES_RIGHT] = 0.0f
        singleFrame[0][TemporalResampler.INDEX_SCORE_RIGHT] = 0.0f
        singleFrame[0][TemporalResampler.INDEX_BOTH_PRESENT] = 0.0f

        val result = resampler.resample(singleFrame, 64)
        assertEquals(64, result.size)
        for (j in 0 until 64) {
            // Left hand coords and score propagated to all frames
            assertEquals(1.0f, result[j][TemporalResampler.INDEX_PRES_LEFT], STRICT_ZERO)
            assertEquals(0.95f, result[j][TemporalResampler.INDEX_SCORE_LEFT], STRICT_ZERO)
            for (c in 0 until 63) {
                assertEquals((c + 1) * 0.01f, result[j][c], 1e-6f)
            }

            // Right hand strictly 0.0f
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_PRES_RIGHT], STRICT_ZERO)
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_SCORE_RIGHT], STRICT_ZERO)
            for (c in 0 until 63) {
                assertEquals(0.0f, result[j][TemporalResampler.SLOT1_START + c], STRICT_ZERO)
            }

            // Interaction strictly 0.0f
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_BOTH_PRESENT], STRICT_ZERO)
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_DIST], STRICT_ZERO)
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_OFFSET_DX], STRICT_ZERO)
            assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_OFFSET_DY], STRICT_ZERO)
        }
    }

    @Test
    fun testSingleActiveDetection_sparsePropagation() {
        // T=10, only frame t=4 is active
        val seq = Array(10) { FloatArray(134) }
        val activeFrame = 4
        for (c in 0 until 63) {
            seq[activeFrame][c] = 0.42f
        }
        seq[activeFrame][TemporalResampler.INDEX_PRES_LEFT] = 1.0f
        seq[activeFrame][TemporalResampler.INDEX_SCORE_LEFT] = 0.88f

        val result = resampler.resample(seq, 64)

        // Find nearest index to activeFrame=4 in target grid
        // targetT = j * 9 / 63 = j / 7. Round(j / 7) == 4 when j in [25..31]
        for (j in 0 until 64) {
            val nearest = kotlin.math.round(j.toFloat() * 9.0f / 63.0f).toInt().coerceIn(0, 9)
            val expectedPres = if (nearest == activeFrame) 1.0f else 0.0f

            assertEquals(expectedPres, result[j][TemporalResampler.INDEX_PRES_LEFT], STRICT_ZERO)
            if (expectedPres > 0.5f) {
                assertEquals(0.88f, result[j][TemporalResampler.INDEX_SCORE_LEFT], 1e-6f)
                for (c in 0 until 63) {
                    assertEquals(0.42f, result[j][c], 1e-6f)
                }
            } else {
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_SCORE_LEFT], STRICT_ZERO)
                for (c in 0 until 63) {
                    assertEquals(0.0f, result[j][c], STRICT_ZERO)
                }
            }
        }
    }

    @Test
    fun testDestinationBufferReuse() {
        val caseObj = casesJson.getJSONObject("case_1_short_sequence")
        val rawInput = parseMatrix(caseObj.getJSONArray("raw_input_134"))
        val expectedResampled = parseMatrix(caseObj.getJSONArray("resampled_64"))

        // Preallocate destination buffer filled with dirty values
        val dest = Array(64) { FloatArray(134) { -999.0f } }

        val returned = resampler.resample(rawInput, 64, dest)
        assertTrue("Must return the exact destination instance", returned === dest)

        verifyParity("destination_buffer_reuse", returned, expectedResampled)
    }

    @Test
    fun testComponentLevel_resampleHandLandmarksAndInteraction() {
        val caseObj = casesJson.getJSONObject("case_6_two_hand_continuous")
        val rawInput = parseMatrix(caseObj.getJSONArray("raw_input_134"))

        val handL = TemporalResampler.resampleHandLandmarks(rawInput, 0, 64)
        val handR = TemporalResampler.resampleHandLandmarks(rawInput, 1, 64)
        val inter = TemporalResampler.resampleInteractionFeatures(rawInput, handL.presence, handR.presence, 64)
        val fused = resampler.resample(rawInput, 64)

        for (j in 0 until 64) {
            // Check left hand
            for (c in 0 until 63) {
                assertEquals(fused[j][c], handL.coords[j][c], STRICT_ZERO)
            }
            assertEquals(fused[j][TemporalResampler.INDEX_PRES_LEFT], handL.presence[j], STRICT_ZERO)
            assertEquals(fused[j][TemporalResampler.INDEX_SCORE_LEFT], handL.score[j], STRICT_ZERO)

            // Check right hand
            for (c in 0 until 63) {
                assertEquals(fused[j][TemporalResampler.SLOT1_START + c], handR.coords[j][c], STRICT_ZERO)
            }
            assertEquals(fused[j][TemporalResampler.INDEX_PRES_RIGHT], handR.presence[j], STRICT_ZERO)
            assertEquals(fused[j][TemporalResampler.INDEX_SCORE_RIGHT], handR.score[j], STRICT_ZERO)

            // Check interaction
            assertEquals(fused[j][TemporalResampler.INDEX_BOTH_PRESENT], inter.bothPresent[j], STRICT_ZERO)
            assertEquals(fused[j][TemporalResampler.INDEX_INTER_DIST], inter.distance[j], STRICT_ZERO)
            assertEquals(fused[j][TemporalResampler.INDEX_INTER_OFFSET_DX], inter.offset[j][0], STRICT_ZERO)
            assertEquals(fused[j][TemporalResampler.INDEX_INTER_OFFSET_DY], inter.offset[j][1], STRICT_ZERO)
        }
    }

    @Test
    fun testInterpHelperFunction() {
        val xp = floatArrayOf(0.0f, 5.0f, 10.0f, 20.0f)
        val fp = floatArrayOf(1.0f, 6.0f, 16.0f, 26.0f)

        // Left boundary / extrapolation
        assertEquals(1.0f, TemporalResampler.interp(-2.0f, xp, fp), 1e-6f)
        assertEquals(1.0f, TemporalResampler.interp(0.0f, xp, fp), 1e-6f)

        // Right boundary / extrapolation
        assertEquals(26.0f, TemporalResampler.interp(20.0f, xp, fp), 1e-6f)
        assertEquals(26.0f, TemporalResampler.interp(25.0f, xp, fp), 1e-6f)

        // Interior linear segments:
        // Segment 0: x in [0, 5], dydx = (6 - 1) / 5 = 1.0 -> y = 1.0 + x
        assertEquals(3.5f, TemporalResampler.interp(2.5f, xp, fp), 1e-6f)

        // Segment 1: x in [5, 10], dydx = (16 - 6) / 5 = 2.0 -> y = 6.0 + 2 * (x - 5)
        assertEquals(10.0f, TemporalResampler.interp(7.0f, xp, fp), 1e-6f)

        // Segment 2: x in [10, 20], dydx = (26 - 16) / 10 = 1.0 -> y = 16.0 + 1 * (x - 10)
        assertEquals(21.0f, TemporalResampler.interp(15.0f, xp, fp), 1e-6f)

        // Degenerate cases
        assertEquals(0.0f, TemporalResampler.interp(1.0f, floatArrayOf(), floatArrayOf()), STRICT_ZERO)
        assertEquals(5.5f, TemporalResampler.interp(10.0f, floatArrayOf(2.0f), floatArrayOf(5.5f)), STRICT_ZERO)
    }

    @Test
    fun testStrictZeroClampingOnInactiveFrames() {
        val caseObj = casesJson.getJSONObject("case_4_sparse_detections")
        val rawInput = parseMatrix(caseObj.getJSONArray("raw_input_134"))
        val result = resampler.resample(rawInput, 64)

        for (j in 0 until 64) {
            val presL = result[j][TemporalResampler.INDEX_PRES_LEFT]
            val presR = result[j][TemporalResampler.INDEX_PRES_RIGHT]
            val both = result[j][TemporalResampler.INDEX_BOTH_PRESENT]

            if (presL <= 0.5f) {
                for (c in 0 until 63) {
                    assertEquals(0.0f, result[j][c], STRICT_ZERO)
                }
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_SCORE_LEFT], STRICT_ZERO)
            }
            if (presR <= 0.5f) {
                for (c in 0 until 63) {
                    assertEquals(0.0f, result[j][TemporalResampler.SLOT1_START + c], STRICT_ZERO)
                }
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_SCORE_RIGHT], STRICT_ZERO)
            }
            if (both <= 0.5f) {
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_DIST], STRICT_ZERO)
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_OFFSET_DX], STRICT_ZERO)
                assertEquals(0.0f, result[j][TemporalResampler.INDEX_INTER_OFFSET_DY], STRICT_ZERO)
            }
        }
    }
}
