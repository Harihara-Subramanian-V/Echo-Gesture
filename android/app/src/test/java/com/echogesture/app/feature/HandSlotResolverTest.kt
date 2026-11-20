package com.echogesture.app.feature

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class HandSlotResolverTest {

    private lateinit var resolver: HandSlotResolver

    @Before
    fun setUp() {
        resolver = HandSlotResolver()
    }

    private fun createLandmarks(wristX: Float, wristY: Float, offset: Float = 0.0f): FloatArray {
        val landmarks = FloatArray(HandSlotResolver.HAND_FEATURE_DIM)
        landmarks[0] = wristX
        landmarks[1] = wristY
        landmarks[2] = 0.0f
        for (i in 1 until HandSlotResolver.LANDMARKS_PER_HAND) {
            landmarks[i * 3 + 0] = wristX + (i * 0.01f) + offset
            landmarks[i * 3 + 1] = wristY + (i * 0.01f) + offset
            landmarks[i * 3 + 2] = 0.0f
        }
        return landmarks
    }

    @Test
    fun testSingleLeftHandResolution() {
        val lms = createLandmarks(wristX = 0.3f, wristY = 0.4f)
        val candidate = DetectedHandCandidate(
            label = "Left",
            score = 0.95f,
            landmarks63 = lms
        )

        val resolved = resolver.resolve(listOf(candidate))

        assertEquals(1.0f, resolved.presLeft, 1e-6f)
        assertEquals(0.0f, resolved.presRight, 1e-6f)
        assertEquals(0.95f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.0f, resolved.scoreRight, 1e-6f)

        assertArrayEquals(lms, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(FloatArray(63), resolved.slot1Landmarks, 1e-6f)

        // Verify tracking cache update
        assertEquals(1.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(0.0f, resolver.prevPresence[1], 1e-6f)
        assertNotNull(resolver.prevWrists)
        assertEquals(0.3f, resolver.prevWrists!![0][0], 1e-6f)
        assertEquals(0.4f, resolver.prevWrists!![0][1], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![1][0], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![1][1], 1e-6f)
    }

    @Test
    fun testSingleRightHandResolution() {
        val lms = createLandmarks(wristX = 0.7f, wristY = 0.6f)
        val candidate = DetectedHandCandidate(
            label = "Right",
            score = 0.88f,
            landmarks63 = lms
        )

        val resolved = resolver.resolve(listOf(candidate))

        assertEquals(0.0f, resolved.presLeft, 1e-6f)
        assertEquals(1.0f, resolved.presRight, 1e-6f)
        assertEquals(0.0f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.88f, resolved.scoreRight, 1e-6f)

        assertArrayEquals(FloatArray(63), resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(lms, resolved.slot1Landmarks, 1e-6f)

        // Verify tracking cache update
        assertEquals(0.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(1.0f, resolver.prevPresence[1], 1e-6f)
        assertNotNull(resolver.prevWrists)
        assertEquals(0.0f, resolver.prevWrists!![0][0], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![0][1], 1e-6f)
        assertEquals(0.7f, resolver.prevWrists!![1][0], 1e-6f)
        assertEquals(0.6f, resolver.prevWrists!![1][1], 1e-6f)
    }

    @Test
    fun testBothHandsDistinctLabels() {
        val leftLms = createLandmarks(wristX = 0.25f, wristY = 0.5f, offset = 0.01f)
        val rightLms = createLandmarks(wristX = 0.75f, wristY = 0.5f, offset = 0.02f)

        val leftCand = DetectedHandCandidate(label = "Left", score = 0.92f, landmarks63 = leftLms)
        val rightCand = DetectedHandCandidate(label = "Right", score = 0.85f, landmarks63 = rightLms)

        // Pass in reverse order (Right first, then Left) to ensure label filtering works correctly
        val resolved = resolver.resolve(listOf(rightCand, leftCand))

        assertEquals(1.0f, resolved.presLeft, 1e-6f)
        assertEquals(1.0f, resolved.presRight, 1e-6f)
        assertEquals(0.92f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.85f, resolved.scoreRight, 1e-6f)

        assertArrayEquals(leftLms, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(rightLms, resolved.slot1Landmarks, 1e-6f)

        assertEquals(1.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(1.0f, resolver.prevPresence[1], 1e-6f)
        assertNotNull(resolver.prevWrists)
        assertEquals(0.25f, resolver.prevWrists!![0][0], 1e-6f)
        assertEquals(0.5f, resolver.prevWrists!![0][1], 1e-6f)
        assertEquals(0.75f, resolver.prevWrists!![1][0], 1e-6f)
        assertEquals(0.5f, resolver.prevWrists!![1][1], 1e-6f)
    }

    @Test
    fun testMultipleCandidatesSelectsHighestConfidence() {
        val leftLowLms = createLandmarks(0.2f, 0.4f, 0.01f)
        val leftHighLms = createLandmarks(0.22f, 0.42f, 0.02f)
        val rightLowLms = createLandmarks(0.8f, 0.4f, 0.03f)
        val rightHighLms = createLandmarks(0.82f, 0.42f, 0.04f)

        val candidates = listOf(
            DetectedHandCandidate("Left", 0.65f, leftLowLms),
            DetectedHandCandidate("Right", 0.70f, rightLowLms),
            DetectedHandCandidate("Left", 0.95f, leftHighLms),
            DetectedHandCandidate("Right", 0.91f, rightHighLms)
        )

        val resolved = resolver.resolve(candidates)

        assertEquals(0.95f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.91f, resolved.scoreRight, 1e-6f)
        assertArrayEquals(leftHighLms, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(rightHighLms, resolved.slot1Landmarks, 1e-6f)
    }

    @Test
    fun testAmbiguousDuplicateLabelsWithoutHistoryHorizontalFallback() {
        // Without previous tracking history, duplicate "Left" labels fallback horizontally:
        // Smaller wristX goes to Slot 0 (viewer-left), larger wristX goes to Slot 1 (viewer-right)
        val rightHandLms = createLandmarks(wristX = 0.72f, wristY = 0.5f)
        val leftHandLms = createLandmarks(wristX = 0.28f, wristY = 0.5f)

        val c0 = DetectedHandCandidate(label = "Left", score = 0.80f, landmarks63 = rightHandLms)
        val c1 = DetectedHandCandidate(label = "Left", score = 0.85f, landmarks63 = leftHandLms)

        val resolved = resolver.resolve(listOf(c0, c1))

        assertEquals(1.0f, resolved.presLeft, 1e-6f)
        assertEquals(1.0f, resolved.presRight, 1e-6f)
        assertEquals(0.85f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.80f, resolved.scoreRight, 1e-6f)

        // Slot 0 got c1 (wristX 0.28), Slot 1 got c0 (wristX 0.72)
        assertArrayEquals(leftHandLms, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(rightHandLms, resolved.slot1Landmarks, 1e-6f)
    }

    @Test
    fun testAmbiguousDuplicateLabelsWithHistoryDirectTracking() {
        // Step 1: Establish valid two-hand history
        val prevLeft = createLandmarks(0.20f, 0.50f)
        val prevRight = createLandmarks(0.80f, 0.50f)
        resolver.resolve(listOf(
            DetectedHandCandidate("Left", 0.9f, prevLeft),
            DetectedHandCandidate("Right", 0.9f, prevRight)
        ))

        // Step 2: Next frame has ambiguous duplicate labels (both "Right")
        // c0 is close to prev Slot 0 (0.21, 0.51)
        // c1 is close to prev Slot 1 (0.79, 0.49)
        val currSlot0 = createLandmarks(0.21f, 0.51f)
        val currSlot1 = createLandmarks(0.79f, 0.49f)

        val c0 = DetectedHandCandidate("Right", 0.88f, currSlot0)
        val c1 = DetectedHandCandidate("Right", 0.87f, currSlot1)

        val resolved = resolver.resolve(listOf(c0, c1))

        // Direct tracking preserved: c0 -> Slot 0, c1 -> Slot 1
        assertArrayEquals(currSlot0, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(currSlot1, resolved.slot1Landmarks, 1e-6f)
        assertEquals(0.88f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.87f, resolved.scoreRight, 1e-6f)
    }

    @Test
    fun testAmbiguousDuplicateLabelsWithHistoryHungarianSwapTracking() {
        // Step 1: Establish valid two-hand history
        // Slot 0: wrist at (0.20, 0.50)
        // Slot 1: wrist at (0.80, 0.50)
        val prevLeft = createLandmarks(0.20f, 0.50f)
        val prevRight = createLandmarks(0.80f, 0.50f)
        resolver.resolve(listOf(
            DetectedHandCandidate("Left", 0.95f, prevLeft),
            DetectedHandCandidate("Right", 0.95f, prevRight)
        ))

        // Step 2: Next frame presents duplicate "Left" labels in reversed order:
        // c0 is at (0.81, 0.51) (near prev Slot 1)
        // c1 is at (0.19, 0.49) (near prev Slot 0)
        val handNearPrevSlot1 = createLandmarks(0.81f, 0.51f)
        val handNearPrevSlot0 = createLandmarks(0.19f, 0.49f)

        val c0 = DetectedHandCandidate("Left", 0.82f, handNearPrevSlot1)
        val c1 = DetectedHandCandidate("Left", 0.86f, handNearPrevSlot0)

        // d_direct = dist(c0, prev0) + dist(c1, prev1) ~ 0.61 + 0.61 = 1.22
        // d_swap   = dist(c1, prev0) + dist(c0, prev1) ~ 0.014 + 0.014 = 0.028
        // d_swap < d_direct -> c1 assigned to Slot 0, c0 assigned to Slot 1!
        val resolved = resolver.resolve(listOf(c0, c1))

        assertArrayEquals(handNearPrevSlot0, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(handNearPrevSlot1, resolved.slot1Landmarks, 1e-6f)
        assertEquals(0.86f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.82f, resolved.scoreRight, 1e-6f)
    }

    @Test
    fun testEmptyCandidatesZeroMasking() {
        val resolved = resolver.resolve(emptyList())

        assertEquals(0.0f, resolved.presLeft, 1e-6f)
        assertEquals(0.0f, resolved.presRight, 1e-6f)
        assertEquals(0.0f, resolved.scoreLeft, 1e-6f)
        assertEquals(0.0f, resolved.scoreRight, 1e-6f)

        assertArrayEquals(FloatArray(63), resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(FloatArray(63), resolved.slot1Landmarks, 1e-6f)

        assertEquals(0.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(0.0f, resolver.prevPresence[1], 1e-6f)
        assertNotNull(resolver.prevWrists)
        assertEquals(0.0f, resolver.prevWrists!![0][0], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![0][1], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![1][0], 1e-6f)
        assertEquals(0.0f, resolver.prevWrists!![1][1], 1e-6f)
    }

    @Test
    fun testResetClearsTemporalTrackingCache() {
        // Establish tracking history
        val prevLeft = createLandmarks(0.20f, 0.50f)
        val prevRight = createLandmarks(0.80f, 0.50f)
        resolver.resolve(listOf(
            DetectedHandCandidate("Left", 0.9f, prevLeft),
            DetectedHandCandidate("Right", 0.9f, prevRight)
        ))

        assertNotNull(resolver.prevWrists)
        assertEquals(1.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(1.0f, resolver.prevPresence[1], 1e-6f)

        // Reset
        resolver.reset()

        assertNull(resolver.prevWrists)
        assertEquals(0.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(0.0f, resolver.prevPresence[1], 1e-6f)

        // Subsequent frame with ambiguous duplicate labels must use horizontal fallback
        val handRight = createLandmarks(0.80f, 0.50f)
        val handLeft = createLandmarks(0.20f, 0.50f)
        val c0 = DetectedHandCandidate("Left", 0.8f, handRight)
        val c1 = DetectedHandCandidate("Left", 0.8f, handLeft)

        val resolved = resolver.resolve(listOf(c0, c1))
        // Horizontal fallback: smaller wristX (handLeft) assigned to Slot 0
        assertArrayEquals(handLeft, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(handRight, resolved.slot1Landmarks, 1e-6f)
    }

    @Test
    fun testSingleHandHistoryDoesNotTriggerHungarianTracking() {
        // Frame 1: Only Left hand was present
        val lms = createLandmarks(0.20f, 0.50f)
        resolver.resolve(listOf(DetectedHandCandidate("Left", 0.9f, lms)))

        assertEquals(1.0f, resolver.prevPresence[0], 1e-6f)
        assertEquals(0.0f, resolver.prevPresence[1], 1e-6f)

        // Frame 2: Ambiguous duplicate labels where c0 is at x=0.75, c1 is at x=0.25
        val handRight = createLandmarks(0.75f, 0.50f)
        val handLeft = createLandmarks(0.25f, 0.50f)
        val c0 = DetectedHandCandidate("Right", 0.8f, handRight)
        val c1 = DetectedHandCandidate("Right", 0.8f, handLeft)

        val resolved = resolver.resolve(listOf(c0, c1))

        // Because prevPresence[1] was 0.0f, Hungarian tracking is skipped and horizontal fallback is used
        assertArrayEquals(handLeft, resolved.slot0Landmarks, 1e-6f)
        assertArrayEquals(handRight, resolved.slot1Landmarks, 1e-6f)
    }

    @Test
    fun testVarargOverload() {
        val lms = createLandmarks(0.3f, 0.4f)
        val cand = DetectedHandCandidate("Left", 0.99f, lms)
        val resolved = resolver.resolve(cand)

        assertEquals(1.0f, resolved.presLeft, 1e-6f)
        assertEquals(0.99f, resolved.scoreLeft, 1e-6f)
        assertArrayEquals(lms, resolved.slot0Landmarks, 1e-6f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testDetectedHandCandidateInvalidSizeThrows() {
        DetectedHandCandidate("Left", 0.9f, FloatArray(10))
    }

    @Test(expected = IllegalArgumentException::class)
    fun testResolvedHandSlotsInvalidSizeThrows() {
        ResolvedHandSlots(
            slot0Landmarks = FloatArray(10),
            slot1Landmarks = FloatArray(63),
            presLeft = 0f,
            presRight = 0f,
            scoreLeft = 0f,
            scoreRight = 0f
        )
    }

    @Test
    fun testDataClassesEqualsAndHashCode() {
        val lms1 = createLandmarks(0.3f, 0.4f)
        val lms2 = createLandmarks(0.3f, 0.4f)
        val lms3 = createLandmarks(0.5f, 0.6f)

        val cand1 = DetectedHandCandidate("Left", 0.9f, lms1)
        val cand2 = DetectedHandCandidate("Left", 0.9f, lms2)
        val cand3 = DetectedHandCandidate("Right", 0.9f, lms3)

        assertEquals(cand1, cand2)
        assertEquals(cand1.hashCode(), cand2.hashCode())
        assertFalse(cand1 == cand3)
        assertTrue(cand1.toString().contains("Left"))

        val slot0A = createLandmarks(0.2f, 0.5f)
        val slot0B = createLandmarks(0.2f, 0.5f)
        val slot1A = createLandmarks(0.8f, 0.5f)
        val slot1B = createLandmarks(0.8f, 0.5f)

        val res1 = ResolvedHandSlots(slot0A, slot1A, 1f, 1f, 0.9f, 0.9f)
        val res2 = ResolvedHandSlots(slot0B, slot1B, 1f, 1f, 0.9f, 0.9f)
        val res3 = ResolvedHandSlots(slot0A, slot1A, 1f, 0f, 0.9f, 0.0f)

        assertEquals(res1, res2)
        assertEquals(res1.hashCode(), res2.hashCode())
        assertFalse(res1 == res3)
        assertTrue(res1.toString().contains("presLeft=1.0"))
    }
}
