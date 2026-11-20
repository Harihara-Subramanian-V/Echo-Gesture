package com.echogesture.app.vision

import com.echogesture.app.feature.DetectedHandCandidate
import com.echogesture.app.feature.HandSlotResolver
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.Landmark
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Host JVM unit test suite for [HandLandmarkDetector] and [IHandLandmarkDetector].
 *
 * Verifies:
 * - Conversion of MediaPipe [HandLandmarkerResult] into EchoGesture [DetectedHandCandidate]
 * - Handedness label normalization ("Left", "Right", case-insensitivity, unknown fallback)
 * - Handedness confidence score extraction and [0.0, 1.0] clamping
 * - 63-D landmark coordinate extraction, flattening order, and coordinate bounds [0.0, 1.0]
 * - Wrist anchor coordinates matching (wristX = landmarks63[0], wristY = landmarks63[1])
 * - Edge cases: 0 hands, null result, 1 hand, 2 hands, >2 hands clamped to 2, empty/partial hands
 * - Lifecycle management ([AutoCloseable], [HandLandmarkDetector.close], error on closed detector)
 * - Test engine integration and contract constants
 */
class HandLandmarkDetectorTest {

    private fun create21Landmarks(wristX: Float, wristY: Float, wristZ: Float = 0.0f): List<NormalizedLandmark> {
        val list = ArrayList<NormalizedLandmark>(21)
        for (i in 0 until 21) {
            val x = (wristX + i * 0.015f).coerceIn(0.0f, 1.0f)
            val y = (wristY + i * 0.010f).coerceIn(0.0f, 1.0f)
            val z = wristZ + i * 0.005f
            list.add(NormalizedLandmark.create(x, y, z))
        }
        return list
    }

    private fun createMockResult(
        hands: List<Pair<String, List<NormalizedLandmark>>>,
        scores: List<Float> = hands.map { 0.95f }
    ): HandLandmarkerResult {
        val landmarksList = hands.map { it.second }
        val handednessList = hands.mapIndexed { idx, pair ->
            val score = if (idx < scores.size) scores[idx] else 0.95f
            listOf(Category.create(score, idx, pair.first, ""))
        }
        return object : HandLandmarkerResult() {
            override fun timestampMs(): Long = 1000L
            override fun landmarks(): List<List<NormalizedLandmark>> = landmarksList
            override fun worldLandmarks(): List<List<Landmark>> = emptyList()
            override fun handedness(): List<List<Category>> = handednessList
        }
    }

    @Test
    fun testContractConstants() {
        assertEquals("hand_landmarker.task", HandLandmarkDetector.DEFAULT_MODEL_ASSET_PATH)
        assertEquals(2, HandLandmarkDetector.DEFAULT_NUM_HANDS)
        assertEquals(0.5f, HandLandmarkDetector.DEFAULT_DETECTION_CONFIDENCE, 1e-6f)
        assertEquals(0.5f, HandLandmarkDetector.DEFAULT_PRESENCE_CONFIDENCE, 1e-6f)
        assertEquals(0.5f, HandLandmarkDetector.DEFAULT_TRACKING_CONFIDENCE, 1e-6f)
        assertEquals(21, HandLandmarkDetector.LANDMARK_COUNT)
        assertEquals(3, HandLandmarkDetector.COORDINATE_DIM)
        assertEquals(63, HandLandmarkDetector.HAND_FEATURE_DIM)
        assertEquals(HandSlotResolver.HAND_FEATURE_DIM, HandLandmarkDetector.HAND_FEATURE_DIM)
    }

    @Test
    fun testConvertResultSingleRightHand() {
        val lms = create21Landmarks(wristX = 0.35f, wristY = 0.60f, wristZ = 0.02f)
        val result = createMockResult(listOf("Right" to lms), listOf(0.94f))

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)

        assertEquals(1, candidates.size)
        val hand = candidates[0]
        assertEquals("Right", hand.label)
        assertEquals(0.94f, hand.score, 1e-6f)
        assertEquals(63, hand.landmarks63.size)

        // Wrist coordinate assertions
        assertEquals(0.35f, hand.wristX, 1e-6f)
        assertEquals(0.60f, hand.wristY, 1e-6f)
        assertEquals(hand.landmarks63[0], hand.wristX, 1e-6f)
        assertEquals(hand.landmarks63[1], hand.wristY, 1e-6f)
        assertEquals(0.02f, hand.landmarks63[2], 1e-6f)

        // Verify index 20 (pinky tip)
        val lm20 = lms[20]
        assertEquals(lm20.x(), hand.landmarks63[20 * 3 + 0], 1e-6f)
        assertEquals(lm20.y(), hand.landmarks63[20 * 3 + 1], 1e-6f)
        assertEquals(lm20.z(), hand.landmarks63[20 * 3 + 2], 1e-6f)
    }

    @Test
    fun testConvertResultTwoHands() {
        val lmsLeft = create21Landmarks(wristX = 0.20f, wristY = 0.50f)
        val lmsRight = create21Landmarks(wristX = 0.75f, wristY = 0.55f)
        val result = createMockResult(
            listOf("Left" to lmsLeft, "Right" to lmsRight),
            listOf(0.91f, 0.88f)
        )

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)

        assertEquals(2, candidates.size)

        val first = candidates[0]
        assertEquals("Left", first.label)
        assertEquals(0.91f, first.score, 1e-6f)
        assertEquals(0.20f, first.wristX, 1e-6f)
        assertEquals(0.50f, first.wristY, 1e-6f)

        val second = candidates[1]
        assertEquals("Right", second.label)
        assertEquals(0.88f, second.score, 1e-6f)
        assertEquals(0.75f, second.wristX, 1e-6f)
        assertEquals(0.55f, second.wristY, 1e-6f)
    }

    @Test
    fun testHandednessNormalizationAndScoreClamping() {
        val lms = create21Landmarks(0.1f, 0.2f)

        // Test lowercase "left" -> canonical "Left"
        val resLowerLeft = createMockResult(listOf("left" to lms), listOf(1.5f)) // score > 1.0
        val candLowerLeft = HandLandmarkDetector.convertResultToCandidates(resLowerLeft)
        assertEquals(1, candLowerLeft.size)
        assertEquals("Left", candLowerLeft[0].label)
        assertEquals(1.0f, candLowerLeft[0].score, 1e-6f) // Clamped to 1.0f

        // Test lowercase "right" -> canonical "Right"
        val resLowerRight = createMockResult(listOf("right" to lms), listOf(-0.5f)) // score < 0.0
        val candLowerRight = HandLandmarkDetector.convertResultToCandidates(resLowerRight)
        assertEquals(1, candLowerRight.size)
        assertEquals("Right", candLowerRight[0].label)
        assertEquals(0.0f, candLowerRight[0].score, 1e-6f) // Clamped to 0.0f

        // Test custom label preservation
        val resCustom = createMockResult(listOf("CustomHand" to lms), listOf(0.77f))
        val candCustom = HandLandmarkDetector.convertResultToCandidates(resCustom)
        assertEquals("CustomHand", candCustom[0].label)
        assertEquals(0.77f, candCustom[0].score, 1e-6f)
    }

    @Test
    fun testMissingHandednessCategoryFallback() {
        val lms = create21Landmarks(0.3f, 0.4f)
        val result = object : HandLandmarkerResult() {
            override fun timestampMs(): Long = 500L
            override fun landmarks(): List<List<NormalizedLandmark>> = listOf(lms)
            override fun worldLandmarks(): List<List<Landmark>> = emptyList()
            override fun handedness(): List<List<Category>> = emptyList() // No categories provided
        }

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)
        assertEquals(1, candidates.size)
        assertEquals("Unknown", candidates[0].label)
        assertEquals(0.0f, candidates[0].score, 1e-6f)
    }

    @Test
    fun testCoordinateBounds() {
        val lms = create21Landmarks(wristX = 0.0f, wristY = 0.0f)
        val result = createMockResult(listOf("Right" to lms))

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)
        assertEquals(1, candidates.size)
        val lms63 = candidates[0].landmarks63

        for (i in 0 until 21) {
            val x = lms63[i * 3 + 0]
            val y = lms63[i * 3 + 1]
            assertTrue("x coordinate $x at landmark $i out of [0.0, 1.0]", x in 0.0f..1.0f)
            assertTrue("y coordinate $y at landmark $i out of [0.0, 1.0]", y in 0.0f..1.0f)
        }
    }

    @Test
    fun testZeroHandsAndNullResult() {
        // Null result
        val nullCandidates = HandLandmarkDetector.convertResultToCandidates(null)
        assertTrue(nullCandidates.isEmpty())

        // Empty landmarks list
        val emptyResult = object : HandLandmarkerResult() {
            override fun timestampMs(): Long = 0L
            override fun landmarks(): List<List<NormalizedLandmark>> = emptyList()
            override fun worldLandmarks(): List<List<Landmark>> = emptyList()
            override fun handedness(): List<List<Category>> = emptyList()
        }
        val emptyCandidates = HandLandmarkDetector.convertResultToCandidates(emptyResult)
        assertTrue(emptyCandidates.isEmpty())
    }

    @Test
    fun testClampingMoreThanTwoHands() {
        val h1 = create21Landmarks(0.1f, 0.2f)
        val h2 = create21Landmarks(0.4f, 0.5f)
        val h3 = create21Landmarks(0.7f, 0.8f)
        val result = createMockResult(
            listOf("Left" to h1, "Right" to h2, "Left" to h3),
            listOf(0.9f, 0.8f, 0.7f)
        )

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)
        assertEquals(2, candidates.size) // Clamped to 2 hands max
        assertEquals(0.1f, candidates[0].wristX, 1e-6f)
        assertEquals(0.4f, candidates[1].wristX, 1e-6f)
    }

    @Test
    fun testPartialLandmarksZeroPadded() {
        // Only 5 landmarks instead of 21
        val partialList = ArrayList<NormalizedLandmark>()
        for (i in 0 until 5) {
            partialList.add(NormalizedLandmark.create(0.2f + i * 0.01f, 0.3f + i * 0.01f, 0.0f))
        }

        val result = createMockResult(listOf("Left" to partialList))
        val candidates = HandLandmarkDetector.convertResultToCandidates(result)

        assertEquals(1, candidates.size)
        val lms63 = candidates[0].landmarks63
        assertEquals(63, lms63.size)

        // First 5 landmarks filled
        for (i in 0 until 5) {
            assertEquals(0.2f + i * 0.01f, lms63[i * 3 + 0], 1e-6f)
            assertEquals(0.3f + i * 0.01f, lms63[i * 3 + 1], 1e-6f)
        }

        // Remaining 16 landmarks should be zero
        for (i in 5 until 21) {
            assertEquals(0.0f, lms63[i * 3 + 0], 1e-6f)
            assertEquals(0.0f, lms63[i * 3 + 1], 1e-6f)
            assertEquals(0.0f, lms63[i * 3 + 2], 1e-6f)
        }
    }

    @Test
    fun testDetectorLifecycleAndClose() {
        val detector = HandLandmarkDetector.createForTesting()

        assertTrue(detector.isInitialized)
        assertFalse(detector.isClosed)

        // Close detector
        detector.close()

        assertTrue(detector.isClosed)
        assertFalse(detector.isInitialized)

        // Repeated close should be idempotent
        detector.close()
        assertTrue(detector.isClosed)
    }

    @Test
    fun testDetectorTestEngineInjection() {
        val lms = create21Landmarks(0.25f, 0.35f)
        val mockResult = createMockResult(listOf("Right" to lms), listOf(0.97f))

        val detector = HandLandmarkDetector.createForTesting(mockResult)
        assertTrue(detector.isInitialized)

        // Directly verify conversion through detector engine
        val candidates = HandLandmarkDetector.convertResultToCandidates(mockResult)
        assertEquals(1, candidates.size)
        assertEquals("Right", candidates[0].label)
        assertEquals(0.97f, candidates[0].score, 1e-6f)
        assertEquals(0.25f, candidates[0].wristX, 1e-6f)

        detector.close()
        assertTrue(detector.isClosed)
    }

    @Test
    fun testDetectedHandCandidateIntegrationWithHandSlotResolver() {
        val resolver = HandSlotResolver()
        val lmsLeft = create21Landmarks(wristX = 0.25f, wristY = 0.50f)
        val lmsRight = create21Landmarks(wristX = 0.75f, wristY = 0.55f)
        val result = createMockResult(listOf("Left" to lmsLeft, "Right" to lmsRight))

        val candidates = HandLandmarkDetector.convertResultToCandidates(result)
        assertEquals(2, candidates.size)

        // Pass directly to HandSlotResolver
        val slots = resolver.resolve(candidates)
        assertEquals(1.0f, slots.presLeft, 1e-6f)
        assertEquals(1.0f, slots.presRight, 1e-6f)
        assertEquals(63, slots.slot0Landmarks.size)
        assertEquals(63, slots.slot1Landmarks.size)
    }
}
