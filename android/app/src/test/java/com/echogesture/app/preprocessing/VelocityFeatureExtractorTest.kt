package com.echogesture.app.preprocessing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Parity and validation test suite for [VelocityFeatureExtractor].
 *
 * Validates 129-D velocity computation, frame 0 boundary conditions (strictly all zeros),
 * transition spike prevention on hand appearance/disappearance, and 263-D concatenation parity
 * against `tests/fixtures/android_parity/stage2_temporal.json`.
 */
class VelocityFeatureExtractorTest {

    private lateinit var rootJson: JSONObject
    private lateinit var casesJson: JSONObject
    private lateinit var extractor: VelocityFeatureExtractor

    data class VelocityParityReport(
        val caseName: String,
        val maxAbsError: Float,
        val meanAbsError: Float,
        val totalElements: Int
    )

    private val parityReports = mutableListOf<VelocityParityReport>()

    @Before
    fun setUp() {
        rootJson = loadStage2Json()
        casesJson = rootJson.getJSONObject("cases")
        extractor = VelocityFeatureExtractor()
    }

    private fun loadStage2Json(): JSONObject {
        val possiblePaths = listOf(
            File("../../tests/fixtures/android_parity/stage2_temporal.json"),
            File("../tests/fixtures/android_parity/stage2_temporal.json"),
            File("tests/fixtures/android_parity/stage2_temporal.json")
        )
        val jsonFile = possiblePaths.firstOrNull { it.exists() }
        assertNotNull(
            "stage2_temporal.json must exist in one of: ${possiblePaths.map { it.absolutePath }}",
            jsonFile
        )
        return JSONObject(jsonFile!!.readText(Charsets.UTF_8))
    }

    private fun parse2DArray(jsonArray: JSONArray): Array<FloatArray> {
        val tCount = jsonArray.length()
        val result = Array(tCount) { FloatArray(0) }
        for (t in 0 until tCount) {
            val row = jsonArray.getJSONArray(t)
            val rowArr = FloatArray(row.length())
            for (c in 0 until row.length()) {
                rowArr[c] = row.getDouble(c).toFloat()
            }
            result[t] = rowArr
        }
        return result
    }

    @Test
    fun testVelocityParityAllCases() {
        val caseNames = listOf(
            "case_1_short_sequence",
            "case_2_full_sequence",
            "case_3_long_sequence",
            "case_4_sparse_detections",
            "case_5_single_hand_continuous",
            "case_6_two_hand_continuous",
            "case_7_alternating_hands",
            "case_8_idle_window"
        )

        val tolerance = 1e-5f

        for (caseName in caseNames) {
            assertTrue("Fixture must contain $caseName", casesJson.has(caseName))
            val caseObj = casesJson.getJSONObject(caseName)

            val standardizedInput = parse2DArray(caseObj.getJSONArray("standardized_64"))
            val expectedVel = parse2DArray(caseObj.getJSONArray("velocity_129"))

            assertEquals("Input length must be 64", 64, standardizedInput.size)
            assertEquals("Expected length must be 64", 64, expectedVel.size)

            val actualVel = extractor.extract(standardizedInput)

            var maxErr = 0.0f
            var sumErr = 0.0
            val totalElements = 64 * 129

            for (t in 0 until 64) {
                assertEquals("Frame $t dimension must be 129", 129, actualVel[t].size)
                for (c in 0 until 129) {
                    val exp = expectedVel[t][c]
                    val act = actualVel[t][c]
                    val err = abs(act - exp)
                    if (err > maxErr) {
                        maxErr = err
                    }
                    sumErr += err
                    assertTrue(
                        "[$caseName] Frame $t Velocity channel $c error $err exceeds tolerance $tolerance (act=$act, exp=$exp)",
                        err <= tolerance
                    )
                }
            }

            val meanErr = (sumErr / totalElements).toFloat()
            parityReports.add(VelocityParityReport(caseName, maxErr, meanErr, totalElements))
            println("Velocity Parity PASS: $caseName - Max Err: $maxErr, Mean Err: $meanErr")
        }

        assertEquals("All 8 cases must be verified", 8, parityReports.size)
    }

    @Test
    fun testFrame0StrictlyZeroAcrossAllCases() {
        val caseNames = listOf(
            "case_1_short_sequence",
            "case_2_full_sequence",
            "case_3_long_sequence",
            "case_4_sparse_detections",
            "case_5_single_hand_continuous",
            "case_6_two_hand_continuous",
            "case_7_alternating_hands",
            "case_8_idle_window"
        )

        for (caseName in caseNames) {
            val caseObj = casesJson.getJSONObject(caseName)
            val standardizedInput = parse2DArray(caseObj.getJSONArray("standardized_64"))
            val actualVel = extractor.extract(standardizedInput)

            for (c in 0 until 129) {
                assertEquals(
                    "[$caseName] Frame 0 velocity channel $c must be strictly 0.0f",
                    0.0f,
                    actualVel[0][c],
                    0.0f
                )
            }
        }
    }

    @Test
    fun testTransitionSpikePreventionCase7AlternatingHands() {
        val caseObj = casesJson.getJSONObject("case_7_alternating_hands")
        val standardizedInput = parse2DArray(caseObj.getJSONArray("standardized_64"))
        val actualVel = extractor.extract(standardizedInput)

        // At frame 32, Left hand disappeared and Right hand appeared.
        // There must be ZERO velocity spikes for both hands.
        for (c in 0 until 63) {
            assertEquals("Case 7 frame 32 Left hand velocity channel $c must be strictly 0.0f",
                0.0f, actualVel[32][c], 0.0f)
        }
        for (c in 0 until 63) {
            assertEquals("Case 7 frame 32 Right hand velocity channel $c must be strictly 0.0f (no appearance spike)",
                0.0f, actualVel[32][63 + c], 0.0f)
        }
        for (c in 0 until 3) {
            assertEquals("Case 7 frame 32 Interaction velocity channel $c must be strictly 0.0f",
                0.0f, actualVel[32][126 + c], 0.0f)
        }
    }

    @Test
    fun testSparseDetectionsSpikePreventionCase4() {
        val caseObj = casesJson.getJSONObject("case_4_sparse_detections")
        val standardizedInput = parse2DArray(caseObj.getJSONArray("standardized_64"))
        val actualVel = extractor.extract(standardizedInput)

        // Isolated active frames must not produce velocity differences with previous inactive frames
        for (t in 1 until 64) {
            val curL = standardizedInput[t][126] > 0.5f
            val prevL = standardizedInput[t - 1][126] > 0.5f
            if (!curL || !prevL) {
                for (c in 0 until 63) {
                    assertEquals("Frame $t Left hand inactive transition velocity must be 0.0f",
                        0.0f, actualVel[t][c], 0.0f)
                }
            }

            val curR = standardizedInput[t][127] > 0.5f
            val prevR = standardizedInput[t - 1][127] > 0.5f
            if (!curR || !prevR) {
                for (c in 0 until 63) {
                    assertEquals("Frame $t Right hand inactive transition velocity must be 0.0f",
                        0.0f, actualVel[t][63 + c], 0.0f)
                }
            }
        }
    }

    @Test
    fun testIdleWindowAllZeroVelocity() {
        val idleCase = casesJson.getJSONObject("case_8_idle_window")
        val standardizedInput = parse2DArray(idleCase.getJSONArray("standardized_64"))
        val actualVel = extractor.extract(standardizedInput)

        for (t in 0 until 64) {
            for (c in 0 until 129) {
                assertEquals("Idle window frame $t velocity channel $c must be strictly 0.0f",
                    0.0f, actualVel[t][c], 0.0f)
            }
        }
    }

    @Test
    fun testSingleHandContinuousVelocity() {
        val singleHandCase = casesJson.getJSONObject("case_5_single_hand_continuous")
        val standardizedInput = parse2DArray(singleHandCase.getJSONArray("standardized_64"))
        val actualVel = extractor.extract(standardizedInput)

        for (t in 0 until 64) {
            // Left hand velocity (0..62) must be strictly 0.0f everywhere
            for (c in 0..62) {
                assertEquals("Left hand velocity frame $t channel $c must be 0.0f", 0.0f, actualVel[t][c], 0.0f)
            }
            // Interaction velocity (126..128) must be strictly 0.0f everywhere
            for (c in 126..128) {
                assertEquals("Interaction velocity frame $t channel $c must be 0.0f", 0.0f, actualVel[t][c], 0.0f)
            }
        }

        // Right hand must have non-zero velocity values for t >= 1
        var rightHasVelocity = false
        for (t in 1 until 64) {
            for (c in 63..125) {
                if (actualVel[t][c] != 0.0f) {
                    rightHasVelocity = true
                    break
                }
            }
        }
        assertTrue("Right hand must exhibit non-zero velocities over time", rightHasVelocity)
    }

    @Test
    fun testConcatenate263DParity() {
        val caseNames = listOf(
            "case_1_short_sequence",
            "case_2_full_sequence",
            "case_3_long_sequence",
            "case_4_sparse_detections",
            "case_5_single_hand_continuous",
            "case_6_two_hand_continuous",
            "case_7_alternating_hands",
            "case_8_idle_window"
        )

        val tolerance = 1e-5f

        for (caseName in caseNames) {
            val caseObj = casesJson.getJSONObject(caseName)
            val standardized = parse2DArray(caseObj.getJSONArray("standardized_64"))
            val expectedVel = parse2DArray(caseObj.getJSONArray("velocity_129"))
            val expected263 = parse2DArray(caseObj.getJSONArray("output_263"))

            val actualVel = extractor.extract(standardized)
            val actual263 = extractor.concatenate(standardized, actualVel)

            assertEquals(64, actual263.size)
            for (t in 0 until 64) {
                assertEquals(263, actual263[t].size)
                for (c in 0 until 263) {
                    val exp = expected263[t][c]
                    val act = actual263[t][c]
                    val err = abs(act - exp)
                    assertTrue(
                        "[$caseName] 263-D tensor mismatch at frame $t channel $c (act=$act, exp=$exp)",
                        err <= tolerance
                    )
                }
            }
        }
    }

    @Test
    fun testDestinationBufferReuse() {
        val caseObj = casesJson.getJSONObject("case_1_short_sequence")
        val standardized = parse2DArray(caseObj.getJSONArray("standardized_64"))

        val preallocatedVel = Array(64) { FloatArray(129) }
        val returnedVel = extractor.extract(standardized, destination = preallocatedVel)
        assertTrue("Must return the exact preallocated instance", returnedVel === preallocatedVel)

        val preallocated263 = Array(64) { FloatArray(263) }
        val returned263 = extractor.concatenate(standardized, returnedVel, destination = preallocated263)
        assertTrue("Must return the exact preallocated 263 instance", returned263 === preallocated263)
    }

    @Test
    fun testEmptySequenceHandling() {
        val emptyInput = Array(0) { FloatArray(134) }
        val result = extractor.extract(emptyInput)
        assertEquals(0, result.size)

        val concatResult = extractor.concatenate(emptyInput, result)
        assertEquals(0, concatResult.size)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testMismatchedDestinationSizeThrows() {
        val input = Array(64) { FloatArray(134) }
        val invalidDest = Array(32) { FloatArray(129) }
        extractor.extract(input, destination = invalidDest)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testMismatchedInputChannelDimThrows() {
        val invalidInput = Array(64) { FloatArray(100) }
        extractor.extract(invalidInput)
    }
}
