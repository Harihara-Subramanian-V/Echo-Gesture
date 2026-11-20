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
 * Parity and validation test suite for [SequenceNormalizer].
 *
 * Validates standardization mathematics, frozen parameter loading, zero-mask preservation,
 * and parity against the Python reference fixture `tests/fixtures/android_parity/stage2_temporal.json`.
 */
class SequenceNormalizerTest {

    private lateinit var rootJson: JSONObject
    private lateinit var casesJson: JSONObject
    private lateinit var normalizer: SequenceNormalizer

    data class ParityReport(
        val caseName: String,
        val maxAbsError: Float,
        val meanAbsError: Float,
        val totalElements: Int
    )

    private val parityReports = mutableListOf<ParityReport>()

    @Before
    fun setUp() {
        rootJson = loadStage2Json()
        casesJson = rootJson.getJSONObject("cases")
        normalizer = SequenceNormalizer.fromJson(rootJson.toString())
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

    private fun loadScalerParamsJson(): File {
        val possiblePaths = listOf(
            File("src/main/assets/scaler_params.json"),
            File("app/src/main/assets/scaler_params.json"),
            File("../../android/app/src/main/assets/scaler_params.json"),
            File("../android/app/src/main/assets/scaler_params.json"),
            File("android/app/src/main/assets/scaler_params.json")
        )
        val file = possiblePaths.firstOrNull { it.exists() }
        assertNotNull(
            "scaler_params.json must exist in one of: ${possiblePaths.map { it.absolutePath }}",
            file
        )
        return file!!
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
    fun testScalerLoadingFromAssetsJson() {
        val assetFile = loadScalerParamsJson()
        val normFromAsset = SequenceNormalizer.fromFile(assetFile)

        assertEquals(134, normFromAsset.mean.size)
        assertEquals(134, normFromAsset.std.size)

        // Presence and handedness channels (126..130) must have mean=0 and std=1
        for (c in 126..130) {
            assertEquals("Mean at channel $c must be 0.0", 0.0f, normFromAsset.mean[c], 1e-6f)
            assertEquals("Std at channel $c must be 1.0", 1.0f, normFromAsset.std[c], 1e-6f)
        }

        // Standard deviations must all be strictly positive
        for (c in 0 until 134) {
            assertTrue("Std at channel $c must be > 0.0", normFromAsset.std[c] > 0.0f)
        }
    }

    @Test
    fun testScalerLoadingFromFixtureNested() {
        assertEquals(134, normalizer.mean.size)
        assertEquals(134, normalizer.std.size)

        val assetFile = loadScalerParamsJson()
        val normFromAsset = SequenceNormalizer.fromFile(assetFile)

        for (c in 0 until 134) {
            assertEquals("Mean channel $c mismatch between fixture and asset",
                normFromAsset.mean[c], normalizer.mean[c], 1e-5f)
            assertEquals("Std channel $c mismatch between fixture and asset",
                normFromAsset.std[c], normalizer.std[c], 1e-5f)
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun testInvalidMeanDimensionThrows() {
        SequenceNormalizer(FloatArray(100), FloatArray(134))
    }

    @Test(expected = IllegalArgumentException::class)
    fun testInvalidStdDimensionThrows() {
        SequenceNormalizer(FloatArray(134), FloatArray(100))
    }

    @Test
    fun testStage2ParityAllCases() {
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

        // Tolerance accounts for 7-decimal JSON export quantization amplified by division by small std values (~0.007)
        val tolerance = 5e-5f

        for (caseName in caseNames) {
            assertTrue("Fixture must contain $caseName", casesJson.has(caseName))
            val caseObj = casesJson.getJSONObject(caseName)

            val resampledInput = parse2DArray(caseObj.getJSONArray("resampled_64"))
            val expectedStd = parse2DArray(caseObj.getJSONArray("standardized_64"))

            assertEquals("Input length must be 64", 64, resampledInput.size)
            assertEquals("Expected length must be 64", 64, expectedStd.size)

            val actualStd = normalizer.normalize(resampledInput)

            var maxErr = 0.0f
            var sumErr = 0.0
            val totalElements = 64 * 134

            for (t in 0 until 64) {
                assertEquals("Frame $t dimension must be 134", 134, actualStd[t].size)
                for (c in 0 until 134) {
                    val exp = expectedStd[t][c]
                    val act = actualStd[t][c]
                    val err = abs(act - exp)
                    if (err > maxErr) {
                        maxErr = err
                    }
                    sumErr += err
                    assertTrue(
                        "[$caseName] Frame $t Channel $c error $err exceeds tolerance $tolerance (act=$act, exp=$exp)",
                        err <= tolerance
                    )
                }
            }

            val meanErr = (sumErr / totalElements).toFloat()
            parityReports.add(ParityReport(caseName, maxErr, meanErr, totalElements))
            println("Parity PASS: $caseName - Max Err: $maxErr, Mean Err: $meanErr")
        }

        assertEquals("All 8 cases must be verified", 8, parityReports.size)
    }

    @Test
    fun testZeroMaskingIdleWindow() {
        val idleCase = casesJson.getJSONObject("case_8_idle_window")
        val resampledInput = parse2DArray(idleCase.getJSONArray("resampled_64"))
        val actualStd = normalizer.normalize(resampledInput)

        for (t in 0 until 64) {
            for (c in 0 until 134) {
                assertEquals("Idle window frame $t channel $c must be strictly 0.0f",
                    0.0f, actualStd[t][c], 0.0f)
            }
        }
    }

    @Test
    fun testZeroMaskingSingleHandContinuous() {
        val singleHandCase = casesJson.getJSONObject("case_5_single_hand_continuous")
        val resampledInput = parse2DArray(singleHandCase.getJSONArray("resampled_64"))
        val actualStd = normalizer.normalize(resampledInput)

        for (t in 0 until 64) {
            // Left hand (0..62) must be strictly 0.0f
            for (c in 0..62) {
                assertEquals("Left hand frame $t channel $c must be 0.0f", 0.0f, actualStd[t][c], 0.0f)
            }
            // Left presence (126) must be 0.0f
            assertEquals(0.0f, actualStd[t][126], 0.0f)

            // Right presence (127) must be 1.0f
            assertEquals(1.0f, actualStd[t][127], 0.0f)

            // Both present (130) must be 0.0f
            assertEquals(0.0f, actualStd[t][130], 0.0f)

            // Interaction features (131..133) must be strictly 0.0f
            for (c in 131..133) {
                assertEquals("Interaction frame $t channel $c must be 0.0f", 0.0f, actualStd[t][c], 0.0f)
            }

            // Right hand (63..125) must have non-zero normalized landmarks
            var rightHasNonZero = false
            for (c in 63..125) {
                if (actualStd[t][c] != 0.0f) {
                    rightHasNonZero = true
                    break
                }
            }
            assertTrue("Right hand must have non-zero normalized landmarks at frame $t", rightHasNonZero)
        }
    }

    @Test
    fun testUnmodifiedChannelsPassThrough() {
        val twoHandCase = casesJson.getJSONObject("case_6_two_hand_continuous")
        val resampledInput = parse2DArray(twoHandCase.getJSONArray("resampled_64"))
        val actualStd = normalizer.normalize(resampledInput)

        for (t in 0 until 64) {
            for (c in 126..130) {
                assertEquals(
                    "Channel $c at frame $t must be unmodified",
                    resampledInput[t][c],
                    actualStd[t][c],
                    0.0f
                )
            }
        }
    }

    @Test
    fun testDestinationBufferReuseAndInPlace() {
        val caseObj = casesJson.getJSONObject("case_1_short_sequence")
        val input = parse2DArray(caseObj.getJSONArray("resampled_64"))

        // 1. Separate destination buffer
        val preallocated = Array(64) { FloatArray(134) }
        val returned = normalizer.normalize(input, destination = preallocated)
        assertTrue("Must return the exact preallocated instance", returned === preallocated)

        // 2. In-place normalization
        val inPlaceInput = Array(64) { t -> input[t].clone() }
        val inPlaceReturned = normalizer.normalize(inPlaceInput, destination = inPlaceInput)
        assertTrue("Must return the same in-place instance", inPlaceReturned === inPlaceInput)

        for (t in 0 until 64) {
            for (c in 0 until 134) {
                assertEquals("In-place normalization mismatch at frame $t channel $c",
                    preallocated[t][c], inPlaceReturned[t][c], 0.0f)
            }
        }
    }

    @Test
    fun testEmptySequenceHandling() {
        val emptyInput = Array(0) { FloatArray(134) }
        val result = normalizer.normalize(emptyInput)
        assertEquals(0, result.size)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testMismatchedDestinationSizeThrows() {
        val input = Array(64) { FloatArray(134) }
        val invalidDest = Array(32) { FloatArray(134) }
        normalizer.normalize(input, destination = invalidDest)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testMismatchedInputChannelDimThrows() {
        val invalidInput = Array(64) { FloatArray(100) }
        normalizer.normalize(invalidInput)
    }
}
