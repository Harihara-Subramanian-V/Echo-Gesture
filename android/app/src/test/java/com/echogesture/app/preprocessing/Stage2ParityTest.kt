package com.echogesture.app.preprocessing

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import kotlin.math.abs

/**
 * Stage 2 Temporal Preprocessing Parity Validation Test Suite for EchoGesture Phase 7B.3.
 *
 * Validates the native Kotlin Stage 2 preprocessing pipeline against all 8 canonical
 * reference fixture cases in `tests/fixtures/android_parity/stage2_temporal.json`.
 *
 * Pipeline Flow:
 * raw_input_134 (T x 134)
 *   -> TemporalResampler -> resampled_64 (64 x 134)
 *   -> SequenceNormalizer -> standardized_64 (64 x 134)
 *   -> VelocityFeatureExtractor -> velocity_129 (64 x 129)
 *   -> Concatenation -> output_263 (64 x 263)
 *
 * Covers all 8 canonical reference cases:
 * 1. case_1_short_sequence: T=24 frames resampled to T=64, normalized, velocity computed.
 * 2. case_2_full_sequence: T=64 frames (identity grid), normalized, velocity computed.
 * 3. case_3_long_sequence: T=96 frames downsampled to T=64, normalized, velocity computed.
 * 4. case_4_sparse_detections: 10 active frames scattered in 50 frames.
 * 5. case_5_single_hand_continuous: Only right hand active throughout 64 frames.
 * 6. case_6_two_hand_continuous: Both hands active throughout 64 frames.
 * 7. case_7_alternating_hands: Left in first half, Right in second half (frame 32 transition spike prevention).
 * 8. case_8_idle_window: All frames empty (all 263 values strictly 0.0).
 */
class Stage2ParityTest {

    data class StageParityStats(
        val stageName: String,
        val maxAbsError: Float,
        val meanAbsError: Float,
        val mismatchCount: Int,
        val totalElements: Int
    )

    data class CaseParityResult(
        val caseName: String,
        val description: String,
        val origT: Int,
        val resampleStats: StageParityStats,
        val normalizeStats: StageParityStats,
        val velocityStats: StageParityStats,
        val concatStats: StageParityStats,
        val endToEndStats: StageParityStats
    )

    companion object {
        private const val FEATURE_DIM_RAW = 134
        private const val FEATURE_DIM_VEL = 129
        private const val FEATURE_DIM_TOTAL = 263
        private const val SEQUENCE_LEN_TARGET = 64

        private const val STAGE_TOLERANCE = 1.0e-5f
        private const val E2E_TOLERANCE = 2.0e-5f

        private lateinit var rootJson: JSONObject
        private lateinit var casesJson: JSONObject
        private lateinit var resampler: TemporalResampler
        private lateinit var normalizer: SequenceNormalizer
        private lateinit var velocityExtractor: VelocityFeatureExtractor

        private val caseResults = mutableListOf<CaseParityResult>()

        @BeforeClass
        @JvmStatic
        fun setUpClass() {
            rootJson = loadFixtureJson()
            casesJson = rootJson.getJSONObject("cases")
            resampler = TemporalResampler(defaultTargetLen = SEQUENCE_LEN_TARGET)

            // Prefer exact frozen scaler params from assets, fallback to embedded fixture scaler
            val scalerFile = listOf(
                File("src/main/assets/scaler_params.json"),
                File("app/src/main/assets/scaler_params.json"),
                File("android/app/src/main/assets/scaler_params.json"),
                File("C:/EchoCam/android/app/src/main/assets/scaler_params.json")
            ).firstOrNull { it.exists() }

            normalizer = if (scalerFile != null) {
                println("Loading frozen scaler parameters from: ${scalerFile.absolutePath}")
                SequenceNormalizer.fromFile(scalerFile)
            } else {
                println("Loading scaler parameters from fixture root JSON")
                SequenceNormalizer.fromJson(rootJson.getJSONObject("scaler").toString())
            }

            velocityExtractor = VelocityFeatureExtractor()
        }

        private fun loadFixtureJson(): JSONObject {
            val possiblePaths = listOf(
                File("../../tests/fixtures/android_parity/stage2_temporal.json"),
                File("../tests/fixtures/android_parity/stage2_temporal.json"),
                File("tests/fixtures/android_parity/stage2_temporal.json"),
                File("C:/EchoCam/tests/fixtures/android_parity/stage2_temporal.json")
            )
            val jsonFile = possiblePaths.firstOrNull { it.exists() }
            assertNotNull(
                "stage2_temporal.json fixture file must exist in one of: ${possiblePaths.map { it.absolutePath }}",
                jsonFile
            )
            println("Loading Stage 2 Parity Fixture from: ${jsonFile!!.absolutePath} (${jsonFile.length() / 1024} KB)")
            return JSONObject(jsonFile.readText(Charsets.UTF_8))
        }

        private fun parse2DArray(jsonArray: JSONArray, expectedCols: Int): Array<FloatArray> {
            val rows = jsonArray.length()
            val result = Array(rows) { FloatArray(expectedCols) }
            for (r in 0 until rows) {
                val rowJson = jsonArray.getJSONArray(r)
                assertEquals("Row $r column count mismatch", expectedCols, rowJson.length())
                val rowArray = result[r]
                for (c in 0 until expectedCols) {
                    rowArray[c] = rowJson.getDouble(c).toFloat()
                }
            }
            return result
        }

        private fun compareTensors(
            caseName: String,
            stageName: String,
            actual: Array<FloatArray>,
            expected: Array<FloatArray>,
            expectedRows: Int,
            expectedCols: Int,
            tolerance: Float
        ): StageParityStats {
            assertEquals("$caseName [$stageName] row count mismatch", expectedRows, actual.size)
            assertEquals("$caseName [$stageName] expected row count mismatch", expectedRows, expected.size)

            var maxErr = 0.0f
            var sumErr = 0.0
            var mismatches = 0
            val totalElements = expectedRows * expectedCols

            for (r in 0 until expectedRows) {
                val actRow = actual[r]
                val expRow = expected[r]
                assertEquals("$caseName [$stageName] row $r col count mismatch", expectedCols, actRow.size)
                assertEquals("$caseName [$stageName] expected row $r col count mismatch", expectedCols, expRow.size)

                for (c in 0 until expectedCols) {
                    val actVal = actRow[c]
                    val expVal = expRow[c]
                    val err = abs(actVal - expVal)
                    if (err > maxErr) {
                        maxErr = err
                    }
                    sumErr += err.toDouble()
                    if (err > tolerance) {
                        mismatches++
                    }
                    assertTrue(
                        "$caseName [$stageName] mismatch at [$r, $c]: act=$actVal, exp=$expVal, err=$err > tol=$tolerance",
                        err <= tolerance
                    )
                }
            }

            val meanErr = (sumErr / totalElements.toDouble()).toFloat()
            return StageParityStats(stageName, maxErr, meanErr, mismatches, totalElements)
        }

        private fun verifyPresenceInvariants(caseName: String, stageName: String, tensor: Array<FloatArray>) {
            for (t in 0 until tensor.size) {
                val frame = tensor[t]
                val pL = frame[126]
                val pR = frame[127]
                val pBoth = frame[130]

                assertTrue(
                    "$caseName [$stageName] frame $t Left Presence (126) must be strictly 0.0f or 1.0f (got $pL)",
                    pL == 0.0f || pL == 1.0f
                )
                assertTrue(
                    "$caseName [$stageName] frame $t Right Presence (127) must be strictly 0.0f or 1.0f (got $pR)",
                    pR == 0.0f || pR == 1.0f
                )
                assertTrue(
                    "$caseName [$stageName] frame $t Both Presence (130) must be strictly 0.0f or 1.0f (got $pBoth)",
                    pBoth == 0.0f || pBoth == 1.0f
                )

                val expectedBoth = if (pL > 0.5f && pR > 0.5f) 1.0f else 0.0f
                assertEquals(
                    "$caseName [$stageName] frame $t Both Presence flag must match (pL && pR)",
                    expectedBoth,
                    pBoth,
                    0.0f
                )
            }
        }

        private fun verifyZeroMaskingInvariants(caseName: String, tensor263: Array<FloatArray>) {
            for (t in 0 until tensor263.size) {
                val frame = tensor263[t]
                val pL = frame[126] > 0.5f
                val pR = frame[127] > 0.5f
                val pBoth = frame[130] > 0.5f

                if (!pL) {
                    for (c in 0 until 63) {
                        assertEquals(
                            "$caseName frame $t Inactive Left Hand coord at $c must be strictly 0.0f",
                            0.0f,
                            frame[c],
                            0.0f
                        )
                    }
                    assertEquals(
                        "$caseName frame $t Inactive Left Hand score at 128 must be strictly 0.0f",
                        0.0f,
                        frame[128],
                        0.0f
                    )
                    for (c in 0 until 63) {
                        assertEquals(
                            "$caseName frame $t Inactive Left Hand velocity at ${134 + c} must be strictly 0.0f",
                            0.0f,
                            frame[134 + c],
                            0.0f
                        )
                    }
                }

                if (!pR) {
                    for (c in 63 until 126) {
                        assertEquals(
                            "$caseName frame $t Inactive Right Hand coord at $c must be strictly 0.0f",
                            0.0f,
                            frame[c],
                            0.0f
                        )
                    }
                    assertEquals(
                        "$caseName frame $t Inactive Right Hand score at 129 must be strictly 0.0f",
                        0.0f,
                        frame[129],
                        0.0f
                    )
                    for (c in 0 until 63) {
                        assertEquals(
                            "$caseName frame $t Inactive Right Hand velocity at ${197 + c} must be strictly 0.0f",
                            0.0f,
                            frame[197 + c],
                            0.0f
                        )
                    }
                }

                if (!pBoth) {
                    for (c in 131 until 134) {
                        assertEquals(
                            "$caseName frame $t Inactive Interaction feature at $c must be strictly 0.0f",
                            0.0f,
                            frame[c],
                            0.0f
                        )
                    }
                    for (c in 0 until 3) {
                        assertEquals(
                            "$caseName frame $t Inactive Interaction velocity at ${260 + c} must be strictly 0.0f",
                            0.0f,
                            frame[260 + c],
                            0.0f
                        )
                    }
                }
            }
        }

        private fun verifyFrame0VelocityInvariant(caseName: String, vel129: Array<FloatArray>) {
            assertEquals("$caseName velocity sequence must have 64 frames", 64, vel129.size)
            for (c in 0 until 129) {
                assertEquals(
                    "$caseName frame 0 velocity channel $c must be strictly 0.0f",
                    0.0f,
                    vel129[0][c],
                    0.0f
                )
            }
        }

        private fun verifyTransitionSpikeInvariant(caseName: String, vel129: Array<FloatArray>) {
            if (caseName == "case_7_alternating_hands") {
                // Frame 32 transition: Left hand disappeared (vel = 0.0f), Right hand appeared (vel = 0.0f)
                for (c in 0 until 63) {
                    assertEquals(
                        "$caseName frame 32 Left hand velocity spike prevention channel $c must be strictly 0.0f",
                        0.0f,
                        vel129[32][c],
                        0.0f
                    )
                }
                for (c in 0 until 63) {
                    assertEquals(
                        "$caseName frame 32 Right hand velocity spike prevention channel $c must be strictly 0.0f",
                        0.0f,
                        vel129[32][63 + c],
                        0.0f
                    )
                }
                for (c in 0 until 3) {
                    assertEquals(
                        "$caseName frame 32 Interaction velocity channel $c must be strictly 0.0f",
                        0.0f,
                        vel129[32][126 + c],
                        0.0f
                    )
                }
            }
        }
    }

    private fun executeCaseParity(caseName: String): CaseParityResult {
        assertTrue("Fixture must contain $caseName", casesJson.has(caseName))
        val caseObj = casesJson.getJSONObject(caseName)
        val description = caseObj.optString("description", "")
        val origT = caseObj.getInt("orig_t")

        val rawInput = parse2DArray(caseObj.getJSONArray("raw_input_134"), FEATURE_DIM_RAW)
        val expectedResampled = parse2DArray(caseObj.getJSONArray("resampled_64"), FEATURE_DIM_RAW)
        val expectedStd = parse2DArray(caseObj.getJSONArray("standardized_64"), FEATURE_DIM_RAW)
        val expectedVel = parse2DArray(caseObj.getJSONArray("velocity_129"), FEATURE_DIM_VEL)
        val expectedOut = parse2DArray(caseObj.getJSONArray("output_263"), FEATURE_DIM_TOTAL)

        assertEquals("$caseName raw input length mismatch", origT, rawInput.size)

        // Stage 2.1: Temporal Resampling
        val actualResampled = resampler.resample(rawInput, SEQUENCE_LEN_TARGET)
        assertEquals("$caseName resampled shape must be [64, 134]", 64, actualResampled.size)
        verifyPresenceInvariants(caseName, "Resampler", actualResampled)
        val resampleStats = compareTensors(
            caseName, "TemporalResampler",
            actualResampled, expectedResampled,
            SEQUENCE_LEN_TARGET, FEATURE_DIM_RAW,
            STAGE_TOLERANCE
        )

        // Stage 2.2: Active Z-Score Normalization
        val actualStd = normalizer.normalize(expectedResampled)
        assertEquals("$caseName normalized shape must be [64, 134]", 64, actualStd.size)
        verifyPresenceInvariants(caseName, "Normalizer", actualStd)
        val normalizeStats = compareTensors(
            caseName, "SequenceNormalizer",
            actualStd, expectedStd,
            SEQUENCE_LEN_TARGET, FEATURE_DIM_RAW,
            STAGE_TOLERANCE
        )

        // Stage 2.3: 129-D Transition-Spike-Free Velocity Feature Extraction
        val actualVel = velocityExtractor.extract(expectedStd)
        assertEquals("$caseName velocity shape must be [64, 129]", 64, actualVel.size)
        verifyFrame0VelocityInvariant(caseName, actualVel)
        verifyTransitionSpikeInvariant(caseName, actualVel)
        val velocityStats = compareTensors(
            caseName, "VelocityFeatureExtractor",
            actualVel, expectedVel,
            SEQUENCE_LEN_TARGET, FEATURE_DIM_VEL,
            STAGE_TOLERANCE
        )

        // Stage 2.4: 263-D Output Concatenation
        val actualConcat = velocityExtractor.concatenate(actualStd, actualVel)
        assertEquals("$caseName concatenated shape must be [64, 263]", 64, actualConcat.size)
        verifyPresenceInvariants(caseName, "Concatenation", actualConcat)
        verifyZeroMaskingInvariants(caseName, actualConcat)
        val concatStats = compareTensors(
            caseName, "263D_Concatenation",
            actualConcat, expectedOut,
            SEQUENCE_LEN_TARGET, FEATURE_DIM_TOTAL,
            STAGE_TOLERANCE
        )

        // Stage 2.5: End-to-End Pipeline Chain
        // raw_input_134 -> resample -> normalize -> velocity -> concatenate -> 263-D
        val e2eResampled = resampler.resample(rawInput, SEQUENCE_LEN_TARGET)
        val e2eStd = normalizer.normalize(e2eResampled)
        val e2eVel = velocityExtractor.extract(e2eStd)
        val e2eOut = velocityExtractor.concatenate(e2eStd, e2eVel)
        assertEquals("$caseName end-to-end shape must be [64, 263]", 64, e2eOut.size)
        verifyPresenceInvariants(caseName, "EndToEnd", e2eOut)
        verifyZeroMaskingInvariants(caseName, e2eOut)
        verifyFrame0VelocityInvariant(caseName, e2eVel)
        verifyTransitionSpikeInvariant(caseName, e2eVel)
        val e2eStats = compareTensors(
            caseName, "EndToEnd_FullChain",
            e2eOut, expectedOut,
            SEQUENCE_LEN_TARGET, FEATURE_DIM_TOTAL,
            E2E_TOLERANCE
        )

        val result = CaseParityResult(
            caseName = caseName,
            description = description,
            origT = origT,
            resampleStats = resampleStats,
            normalizeStats = normalizeStats,
            velocityStats = velocityStats,
            concatStats = concatStats,
            endToEndStats = e2eStats
        )
        caseResults.add(result)

        println(
            "=== Parity PASS: %-30s (T_in=%2d) | Resample: %e | Normal: %e | Velocity: %e | Output263: %e | E2E: %e ===".format(
                caseName, origT,
                resampleStats.maxAbsError,
                normalizeStats.maxAbsError,
                velocityStats.maxAbsError,
                concatStats.maxAbsError,
                e2eStats.maxAbsError
            )
        )

        return result
    }

    @Test
    fun testCase1_ShortSequence() {
        val result = executeCaseParity("case_1_short_sequence")
        assertEquals(24, result.origT)
    }

    @Test
    fun testCase2_FullSequence() {
        val result = executeCaseParity("case_2_full_sequence")
        assertEquals(64, result.origT)
    }

    @Test
    fun testCase3_LongSequence() {
        val result = executeCaseParity("case_3_long_sequence")
        assertEquals(96, result.origT)
    }

    @Test
    fun testCase4_SparseDetections() {
        val result = executeCaseParity("case_4_sparse_detections")
        assertEquals(50, result.origT)
    }

    @Test
    fun testCase5_SingleHandContinuous() {
        val result = executeCaseParity("case_5_single_hand_continuous")
        assertEquals(64, result.origT)
    }

    @Test
    fun testCase6_TwoHandContinuous() {
        val result = executeCaseParity("case_6_two_hand_continuous")
        assertEquals(64, result.origT)
    }

    @Test
    fun testCase7_AlternatingHands() {
        val result = executeCaseParity("case_7_alternating_hands")
        assertEquals(64, result.origT)
    }

    @Test
    fun testCase8_IdleWindow() {
        val result = executeCaseParity("case_8_idle_window")
        assertEquals(64, result.origT)

        // Idle window must be strictly 0.0 across all 64 x 263 elements
        val rawInput = parse2DArray(casesJson.getJSONObject("case_8_idle_window").getJSONArray("raw_input_134"), FEATURE_DIM_RAW)
        val e2eResampled = resampler.resample(rawInput, SEQUENCE_LEN_TARGET)
        val e2eStd = normalizer.normalize(e2eResampled)
        val e2eVel = velocityExtractor.extract(e2eStd)
        val e2eOut = velocityExtractor.concatenate(e2eStd, e2eVel)

        for (t in 0 until SEQUENCE_LEN_TARGET) {
            for (c in 0 until FEATURE_DIM_TOTAL) {
                assertEquals(
                    "Idle window frame $t channel $c must be strictly 0.0f",
                    0.0f,
                    e2eOut[t][c],
                    0.0f
                )
            }
        }
    }

    @Test
    fun testOverallStage2ParitySummary() {
        // Run any cases not yet executed (or verify all 8)
        val allCaseNames = listOf(
            "case_1_short_sequence",
            "case_2_full_sequence",
            "case_3_long_sequence",
            "case_4_sparse_detections",
            "case_5_single_hand_continuous",
            "case_6_two_hand_continuous",
            "case_7_alternating_hands",
            "case_8_idle_window"
        )
        for (name in allCaseNames) {
            if (caseResults.none { it.caseName == name }) {
                executeCaseParity(name)
            }
        }

        assertEquals("All 8 cases must have been executed", 8, caseResults.size)

        var globalMaxResample = 0.0f
        var globalMaxNorm = 0.0f
        var globalMaxVel = 0.0f
        var globalMaxConcat = 0.0f
        var globalMaxE2E = 0.0f

        var totalMismatches = 0
        var totalElementsEvaluated = 0

        println("\n" + "=".repeat(110))
        println("                           STAGE 2 TEMPORAL PREPROCESSING PARITY VERIFICATION REPORT")
        println("=".repeat(110))
        println(
            "%-30s | %-5s | %-12s | %-12s | %-12s | %-12s | %-12s".format(
                "Case Name", "T_in", "ResampleErr", "NormErr", "VelErr", "ConcatErr", "E2E Err"
            )
        )
        println("-".repeat(110))

        for (res in caseResults) {
            println(
                "%-30s | %-5d | %-12e | %-12e | %-12e | %-12e | %-12e".format(
                    res.caseName,
                    res.origT,
                    res.resampleStats.maxAbsError,
                    res.normalizeStats.maxAbsError,
                    res.velocityStats.maxAbsError,
                    res.concatStats.maxAbsError,
                    res.endToEndStats.maxAbsError
                )
            )

            if (res.resampleStats.maxAbsError > globalMaxResample) globalMaxResample = res.resampleStats.maxAbsError
            if (res.normalizeStats.maxAbsError > globalMaxNorm) globalMaxNorm = res.normalizeStats.maxAbsError
            if (res.velocityStats.maxAbsError > globalMaxVel) globalMaxVel = res.velocityStats.maxAbsError
            if (res.concatStats.maxAbsError > globalMaxConcat) globalMaxConcat = res.concatStats.maxAbsError
            if (res.endToEndStats.maxAbsError > globalMaxE2E) globalMaxE2E = res.endToEndStats.maxAbsError

            totalMismatches += res.resampleStats.mismatchCount + res.normalizeStats.mismatchCount +
                    res.velocityStats.mismatchCount + res.concatStats.mismatchCount + res.endToEndStats.mismatchCount
            totalElementsEvaluated += res.resampleStats.totalElements + res.normalizeStats.totalElements +
                    res.velocityStats.totalElements + res.concatStats.totalElements + res.endToEndStats.totalElements
        }

        println("-".repeat(110))
        println("OVERALL MAXIMUM ABSOLUTE ERRORS:")
        println("  TemporalResampler (134-D):       %e (tolerance: %e)".format(globalMaxResample, STAGE_TOLERANCE))
        println("  SequenceNormalizer (134-D):      %e (tolerance: %e)".format(globalMaxNorm, STAGE_TOLERANCE))
        println("  VelocityFeatureExtractor (129-D):%e (tolerance: %e)".format(globalMaxVel, STAGE_TOLERANCE))
        println("  263-D Concatenation:             %e (tolerance: %e)".format(globalMaxConcat, STAGE_TOLERANCE))
        println("  End-to-End Pipeline Chain:       %e (tolerance: %e)".format(globalMaxE2E, E2E_TOLERANCE))
        println("TOTAL MISMATCHES:                  $totalMismatches / $totalElementsEvaluated elements")
        println("=".repeat(110) + "\n")

        assertTrue("Resampling max error must be <= $STAGE_TOLERANCE", globalMaxResample <= STAGE_TOLERANCE)
        assertTrue("Normalization max error must be <= $STAGE_TOLERANCE", globalMaxNorm <= STAGE_TOLERANCE)
        assertTrue("Velocity extraction max error must be <= $STAGE_TOLERANCE", globalMaxVel <= STAGE_TOLERANCE)
        assertTrue("Concatenation max error must be <= $STAGE_TOLERANCE", globalMaxConcat <= STAGE_TOLERANCE)
        assertTrue("End-to-end max error must be <= $E2E_TOLERANCE", globalMaxE2E <= E2E_TOLERANCE)
        assertEquals("Total mismatched elements across all evaluated stages must be 0", 0, totalMismatches)
    }
}
