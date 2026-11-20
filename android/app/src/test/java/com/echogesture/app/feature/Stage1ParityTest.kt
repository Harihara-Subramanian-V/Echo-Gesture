package com.echogesture.app.feature

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
 * Stage 1 Feature Parity Validation Test Suite for EchoGesture Phase 7B.2.
 *
 * Validates the Kotlin 134-D feature extraction implementation against the Stage 1
 * Python reference fixtures in `tests/fixtures/android_parity/stage1_features.json`.
 *
 * Covers all 7 canonical Stage 1 cases:
 * 1. case_1_left_only: Left hand only (right slot zero-masked).
 * 2. case_2_right_only: Right hand only (left slot zero-masked).
 * 3. case_3_both_hands: Both hands present in 640x480 frame.
 * 4. case_4_missing_hands: Neither hand detected (strictly all 134 values zero).
 * 5. case_5_transition_sequence: 5-frame temporal sequence with appearance/motion/disappearance.
 * 6. case_6_hungarian_tracking: 2-frame ambiguous duplicate label sequence verifying Hungarian swap resolution.
 * 7. case_7_aspect_ratios: Multi-resolution verification (640x480, 1280x720, 1920x1080).
 */
class Stage1ParityTest {

    private lateinit var rootJson: JSONObject
    private lateinit var casesJson: JSONObject
    private lateinit var resolver: HandSlotResolver
    private lateinit var extractor: FeatureExtractor134D

    data class ParityStats(
        val caseName: String,
        val maxAbsError: Float,
        val meanAbsError: Float,
        val mismatchCount: Int,
        val totalElements: Int
    )

    private val allStats = mutableListOf<ParityStats>()

    @Before
    fun setUp() {
        rootJson = loadJson()
        casesJson = rootJson.getJSONObject("cases")
        resolver = HandSlotResolver()
        extractor = FeatureExtractor134D(preallocateOutput = false)
    }

    private fun loadJson(): JSONObject {
        val possiblePaths = listOf(
            File("src/test/resources/stage1_features.json"),
            File("../../tests/fixtures/android_parity/stage1_features.json"),
            File("../tests/fixtures/android_parity/stage1_features.json"),
            File("tests/fixtures/android_parity/stage1_features.json")
        )
        val jsonFile = possiblePaths.firstOrNull { it.exists() }
        assertNotNull(
            "stage1_features.json fixture file must exist in one of: ${possiblePaths.map { it.absolutePath }}",
            jsonFile
        )
        return JSONObject(jsonFile!!.readText(Charsets.UTF_8))
    }

    private fun parseLandmarks63(jsonArray: JSONArray): FloatArray {
        assertEquals("Landmarks array must contain 21 points", 21, jsonArray.length())
        val result = FloatArray(63)
        var idx = 0
        for (i in 0 until jsonArray.length()) {
            val point = jsonArray.getJSONArray(i)
            assertEquals("Each point must have 3 coordinates (x, y, z)", 3, point.length())
            result[idx++] = point.getDouble(0).toFloat()
            result[idx++] = point.getDouble(1).toFloat()
            result[idx++] = point.getDouble(2).toFloat()
        }
        return result
    }

    private fun parseExpected134(jsonArray: JSONArray): FloatArray {
        assertEquals("Feature array must have length 134", 134, jsonArray.length())
        val result = FloatArray(134)
        for (i in 0 until 134) {
            result[i] = jsonArray.getDouble(i).toFloat()
        }
        return result
    }

    private fun verifyParity(
        caseName: String,
        actual: FloatArray,
        expected: FloatArray,
        tolerance: Float = 1e-6f
    ): ParityStats {
        assertEquals("$caseName: actual feature vector size must be 134", 134, actual.size)
        assertEquals("$caseName: expected feature vector size must be 134", 134, expected.size)

        // 1. Exact equality for binary presence and co-presence flags
        assertEquals(
            "$caseName: Left hand presence flag (features[126]) must match exactly",
            expected[126],
            actual[126],
            0.0f
        )
        assertEquals(
            "$caseName: Right hand presence flag (features[127]) must match exactly",
            expected[127],
            actual[127],
            0.0f
        )
        assertEquals(
            "$caseName: Both hands co-presence flag (features[130]) must match exactly",
            expected[130],
            actual[130],
            0.0f
        )

        // 2. Exact equality for inactive hand zero-masking
        if (expected[126] == 0.0f) {
            for (i in 0 until 63) {
                assertEquals("$caseName: Inactive slot0 landmark coordinate at index $i must be strictly 0.0f", 0.0f, actual[i], 0.0f)
            }
            assertEquals("$caseName: Inactive slot0 confidence score at index 128 must be strictly 0.0f", 0.0f, actual[128], 0.0f)
        }
        if (expected[127] == 0.0f) {
            for (i in 63 until 126) {
                assertEquals("$caseName: Inactive slot1 landmark coordinate at index $i must be strictly 0.0f", 0.0f, actual[i], 0.0f)
            }
            assertEquals("$caseName: Inactive slot1 confidence score at index 129 must be strictly 0.0f", 0.0f, actual[129], 0.0f)
        }
        if (expected[130] == 0.0f) {
            assertEquals("$caseName: Inactive inter-hand distance at index 131 must be strictly 0.0f", 0.0f, actual[131], 0.0f)
            assertEquals("$caseName: Inactive inter-hand offset dx at index 132 must be strictly 0.0f", 0.0f, actual[132], 0.0f)
            assertEquals("$caseName: Inactive inter-hand offset dy at index 133 must be strictly 0.0f", 0.0f, actual[133], 0.0f)
        }

        // 3. Numerical tolerance and metric calculation
        var maxErr = 0.0f
        var sumErr = 0.0
        var mismatches = 0

        for (i in 0 until 134) {
            val err = abs(actual[i] - expected[i])
            if (err > maxErr) {
                maxErr = err
            }
            sumErr += err.toDouble()
            if (err > tolerance) {
                mismatches++
            }
            assertEquals(
                "$caseName: Value at index $i mismatch (expected=${expected[i]}, actual=${actual[i]}, absErr=$err, tol=$tolerance)",
                expected[i],
                actual[i],
                tolerance
            )
        }

        val meanErr = (sumErr / 134.0).toFloat()
        val stats = ParityStats(caseName, maxErr, meanErr, mismatches, 134)
        allStats.add(stats)
        println("Parity [${caseName.padEnd(30)}] -> MaxAbsError: %e, MeanAbsError: %e, Mismatches: %d / 134 (tol=%e)".format(
            maxErr, meanErr, mismatches, tolerance
        ))
        return stats
    }

    @Test
    fun testCase1_LeftOnly() {
        val caseData = casesJson.getJSONObject("case_1_left_only")
        val breakdown = caseData.getJSONObject("breakdown")
        val expected = parseExpected134(caseData.getJSONArray("features_134"))

        val leftLms = parseLandmarks63(breakdown.getJSONArray("slot0_left_landmarks"))
        val leftScore = breakdown.getDouble("left_score").toFloat()

        val candidate = DetectedHandCandidate(
            label = "Left",
            score = leftScore,
            landmarks63 = leftLms
        )

        resolver.reset()
        val slots = resolver.resolve(listOf(candidate))
        val actual = extractor.extract(slots, 640, 480)

        verifyParity("case_1_left_only", actual, expected)
    }

    @Test
    fun testCase2_RightOnly() {
        val caseData = casesJson.getJSONObject("case_2_right_only")
        val breakdown = caseData.getJSONObject("breakdown")
        val expected = parseExpected134(caseData.getJSONArray("features_134"))

        val rightLms = parseLandmarks63(breakdown.getJSONArray("slot1_right_landmarks"))
        val rightScore = breakdown.getDouble("right_score").toFloat()

        val candidate = DetectedHandCandidate(
            label = "Right",
            score = rightScore,
            landmarks63 = rightLms
        )

        resolver.reset()
        val slots = resolver.resolve(listOf(candidate))
        val actual = extractor.extract(slots, 640, 480)

        verifyParity("case_2_right_only", actual, expected)
    }

    @Test
    fun testCase3_BothHands() {
        val caseData = casesJson.getJSONObject("case_3_both_hands")
        val breakdown = caseData.getJSONObject("breakdown")
        val expected = parseExpected134(caseData.getJSONArray("features_134"))

        val leftLms = parseLandmarks63(breakdown.getJSONArray("slot0_left_landmarks"))
        val leftScore = breakdown.getDouble("left_score").toFloat()
        val rightLms = parseLandmarks63(breakdown.getJSONArray("slot1_right_landmarks"))
        val rightScore = breakdown.getDouble("right_score").toFloat()

        val leftCandidate = DetectedHandCandidate(
            label = "Left",
            score = leftScore,
            landmarks63 = leftLms
        )
        val rightCandidate = DetectedHandCandidate(
            label = "Right",
            score = rightScore,
            landmarks63 = rightLms
        )

        resolver.reset()
        val slots = resolver.resolve(listOf(leftCandidate, rightCandidate))
        val actual = extractor.extract(slots, 640, 480)

        verifyParity("case_3_both_hands", actual, expected)
    }

    @Test
    fun testCase4_MissingHands() {
        val caseData = casesJson.getJSONObject("case_4_missing_hands")
        val expected = parseExpected134(caseData.getJSONArray("features_134"))

        resolver.reset()
        val slots = resolver.resolve(emptyList())
        val actual = extractor.extract(slots, 640, 480)

        // Neither hand detected: all 134 values strictly zero
        for (i in 0 until 134) {
            assertEquals("Index $i in missing hands case must be strictly 0.0f", 0.0f, actual[i], 0.0f)
        }

        verifyParity("case_4_missing_hands", actual, expected)
    }

    @Test
    fun testCase5_TransitionSequence() {
        val caseData = casesJson.getJSONObject("case_5_transition_sequence")
        val framesArray = caseData.getJSONArray("frames")
        assertEquals(5, framesArray.length())

        resolver.reset()

        for (fIdx in 0 until framesArray.length()) {
            val frameObj = framesArray.getJSONObject(fIdx)
            val breakdown = frameObj.getJSONObject("breakdown")
            val expected = parseExpected134(frameObj.getJSONArray("features_134"))

            val presL = breakdown.getDouble("left_presence").toFloat()
            val presR = breakdown.getDouble("right_presence").toFloat()

            val candidates = mutableListOf<DetectedHandCandidate>()
            if (presL > 0.5f) {
                val lmsL = parseLandmarks63(breakdown.getJSONArray("slot0_left_landmarks"))
                val scoreL = breakdown.getDouble("left_score").toFloat()
                candidates.add(DetectedHandCandidate("Left", scoreL, lmsL))
            }
            if (presR > 0.5f) {
                val lmsR = parseLandmarks63(breakdown.getJSONArray("slot1_right_landmarks"))
                val scoreR = breakdown.getDouble("right_score").toFloat()
                candidates.add(DetectedHandCandidate("Right", scoreR, lmsR))
            }

            val slots = resolver.resolve(candidates)
            val actual = extractor.extract(slots, 640, 480)

            verifyParity("case_5_transition_frame_$fIdx", actual, expected)
        }
    }

    @Test
    fun testCase6_HungarianTracking() {
        val caseData = casesJson.getJSONObject("case_6_hungarian_tracking")
        val framesArray = caseData.getJSONArray("frames")
        assertEquals(2, framesArray.length())

        resolver.reset()

        // Frame 0: Standard Left and Right initialization
        val frame0 = framesArray.getJSONObject(0)
        val b0 = frame0.getJSONObject("breakdown")
        val expected0 = parseExpected134(frame0.getJSONArray("features_134"))

        val candL0 = DetectedHandCandidate(
            label = "Left",
            score = 0.95f,
            landmarks63 = parseLandmarks63(b0.getJSONArray("slot0_left_landmarks"))
        )
        val candR0 = DetectedHandCandidate(
            label = "Right",
            score = 0.95f,
            landmarks63 = parseLandmarks63(b0.getJSONArray("slot1_right_landmarks"))
        )

        val slots0 = resolver.resolve(listOf(candL0, candR0))
        val actual0 = extractor.extract(slots0, 640, 480)
        verifyParity("case_6_hungarian_frame_0", actual0, expected0)

        // Frame 1: Ambiguous duplicate 'Left' labels
        // Candidate 0: label="Left", score=0.88, wrist at (0.74, 0.51) -> corresponds to slot1 (Viewer-Right)
        // Candidate 1: label="Left", score=0.91, wrist at (0.26, 0.49) -> corresponds to slot0 (Viewer-Left)
        val frame1 = framesArray.getJSONObject(1)
        val b1 = frame1.getJSONObject("breakdown")
        val expected1 = parseExpected134(frame1.getJSONArray("features_134"))

        val candDup0 = DetectedHandCandidate(
            label = "Left",
            score = 0.88f,
            landmarks63 = parseLandmarks63(b1.getJSONArray("slot1_right_landmarks"))
        )
        val candDup1 = DetectedHandCandidate(
            label = "Left",
            score = 0.91f,
            landmarks63 = parseLandmarks63(b1.getJSONArray("slot0_left_landmarks"))
        )

        // Passing candidate 0 then candidate 1 forces Hungarian nearest-wrist swap
        val slots1 = resolver.resolve(listOf(candDup0, candDup1))
        val actual1 = extractor.extract(slots1, 640, 480)
        verifyParity("case_6_hungarian_frame_1", actual1, expected1)

        // Verify Hungarian tracking correctly mapped candDup1 -> Slot 0 and candDup0 -> Slot 1
        assertEquals(0.91f, slots1.scoreLeft, 1e-6f)
        assertEquals(0.88f, slots1.scoreRight, 1e-6f)
    }

    @Test
    fun testCase7_AspectRatios() {
        val caseData = casesJson.getJSONObject("case_7_aspect_ratios")
        val resolutions = caseData.getJSONObject("resolutions")

        val resKeys = listOf("640x480", "1280x720", "1920x1080")
        val actualVectors = mutableMapOf<String, FloatArray>()

        for (resKey in resKeys) {
            val resObj = resolutions.getJSONObject(resKey)
            val w = resObj.getInt("width")
            val h = resObj.getInt("height")
            val breakdown = resObj.getJSONObject("breakdown")
            val expected = parseExpected134(resObj.getJSONArray("features_134"))

            val leftLms = parseLandmarks63(breakdown.getJSONArray("slot0_left_landmarks"))
            val rightLms = parseLandmarks63(breakdown.getJSONArray("slot1_right_landmarks"))
            val leftScore = breakdown.getDouble("left_score").toFloat()
            val rightScore = breakdown.getDouble("right_score").toFloat()

            val candLeft = DetectedHandCandidate("Left", leftScore, leftLms)
            val candRight = DetectedHandCandidate("Right", rightScore, rightLms)

            resolver.reset()
            val slots = resolver.resolve(listOf(candLeft, candRight))
            val actual = extractor.extract(slots, w, h)

            actualVectors[resKey] = actual
            verifyParity("case_7_aspect_ratio_$resKey", actual, expected)
        }

        // Cross-resolution invariance check:
        // 1280x720 and 1920x1080 have identical aspect ratio (16:9), so geometric features must match
        val feat1280 = actualVectors["1280x720"]!!
        val feat1920 = actualVectors["1920x1080"]!!

        assertEquals(
            "Inter-hand distance must be invariant between 1280x720 and 1920x1080",
            feat1280[131],
            feat1920[131],
            1e-6f
        )
        assertEquals(
            "Inter-hand offset dx must be invariant between 1280x720 and 1920x1080",
            feat1280[132],
            feat1920[132],
            1e-6f
        )
        assertEquals(
            "Inter-hand offset dy must be invariant between 1280x720 and 1920x1080",
            feat1280[133],
            feat1920[133],
            1e-6f
        )
    }
}
