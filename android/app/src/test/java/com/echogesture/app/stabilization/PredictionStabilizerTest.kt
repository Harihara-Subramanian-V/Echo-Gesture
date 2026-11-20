package com.echogesture.app.stabilization

import com.echogesture.app.inference.PredictionCandidate
import com.echogesture.app.inference.PredictionResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Comprehensive unit test suite for [PredictionStabilizer].
 *
 * Validates:
 * 1. 6-state state machine transitions:
 *    WAITING_FOR_HANDS -> COLLECTING_FRAMES -> RECOGNIZING -> ACTIVE_PREDICTION -> TRACKING_LOST -> IDLE.
 * 2. Anti-flicker hysteresis and weighted voting (2-frame consensus requirement).
 * 3. Single-frame spike rejection for low/moderate confidence fluctuations.
 * 4. High-confidence fast switching (score >= 2x threshold = 0.70f).
 * 5. Tracking lost tolerance (1 to 11 empty frames retains sign, restores on hand recovery).
 * 6. Idle dropout after 12 empty frames with complete sign clearing.
 * 7. Stale prediction decay when predictions become uncertain.
 * 8. Integration with [PredictionResult] and [ClassPrediction] mappings.
 */
class PredictionStabilizerTest {

    private lateinit var stabilizer: PredictionStabilizer

    @Before
    fun setUp() {
        stabilizer = PredictionStabilizer(
            windowSize = 5,
            confidenceThreshold = 0.35f,
            idleDropoutFrames = 12,
            uncertainDropoutFrames = 5,
            minFramesForInference = 16
        )
    }

    private fun mockCandidate(rank: Int, classId: Int, name: String, conf: Float): PredictionCandidate {
        val qByte = ((conf * 256.0f) - 128).toInt().coerceIn(-128, 127).toByte()
        return PredictionCandidate(rank, classId, name, qByte, conf)
    }

    private fun mockPrediction(name: String, classId: Int, conf: Float): PredictionResult {
        val cand1 = mockCandidate(1, classId, name, conf)
        val cand2 = mockCandidate(2, (classId + 1) % 40, "alt_1", (1.0f - conf) * 0.5f)
        val cand3 = mockCandidate(3, (classId + 2) % 40, "alt_2", (1.0f - conf) * 0.3f)
        val cand4 = mockCandidate(4, (classId + 3) % 40, "alt_3", 0.05f)
        val cand5 = mockCandidate(5, (classId + 4) % 40, "alt_4", 0.05f)
        return PredictionResult(
            topClassId = classId,
            topClassName = name,
            topConfidence = conf,
            topQuantizedScore = cand1.quantizedScore,
            top5 = listOf(cand1, cand2, cand3, cand4, cand5),
            allProbabilities = FloatArray(40) { 0.025f }
        )
    }

    @Test
    fun testInitialState() {
        val state = stabilizer.currentState()
        assertEquals(RecognitionState.WAITING_FOR_HANDS, state.state)
        assertEquals("IDLE", state.recognizedSign)
        assertEquals(-1, state.classId)
        assertEquals(0.0f, state.confidence, 1e-6f)
        assertTrue(state.top3.isEmpty())
        assertEquals(0, state.consecutiveIdleFrames)
        assertFalse(state.isStable)
    }

    @Test
    fun testCollectingFramesTransitionAndDropout() {
        // Hands appear, but buffer has fewer than 16 frames
        for (f in 1..15) {
            val s = stabilizer.update(
                predictedClass = "hello",
                predictedClassId = 5,
                confidence = 0.60f,
                hasHands = true,
                bufferFrameCount = f
            )
            assertEquals(RecognitionState.COLLECTING_FRAMES, s.state)
            assertEquals("IDLE", s.recognizedSign)
            assertEquals(-1, s.classId)
            assertFalse(s.isStable)
        }

        // Hands disappear before reaching 16 frames -> transitions back to WAITING_FOR_HANDS
        val lost = stabilizer.update(
            predictedClass = "hello",
            predictedClassId = 5,
            confidence = 0.0f,
            hasHands = false,
            bufferFrameCount = 0
        )
        assertEquals(RecognitionState.WAITING_FOR_HANDS, lost.state)
        assertEquals(1, lost.consecutiveIdleFrames)
    }

    @Test
    fun testRecognizingStateWhenUncertain() {
        // Buffer has reached 16 frames, but model confidence is below threshold (0.35f)
        for (i in 0 until 3) {
            val s = stabilizer.update(
                predictedClass = "hello",
                predictedClassId = 5,
                confidence = 0.20f, // below 0.35
                hasHands = true,
                bufferFrameCount = 16 + i
            )
            assertEquals(RecognitionState.RECOGNIZING, s.state)
            assertEquals("IDLE", s.recognizedSign)
            assertFalse(s.isStable)
        }
    }

    @Test
    fun testActivePredictionTwoFrameConsensus() {
        // Frame 1 with confidence 0.50f (0.50 < 0.35 * 1.5 = 0.525, votes = 1)
        val f1 = stabilizer.update(
            predictedClass = "hello",
            predictedClassId = 5,
            confidence = 0.50f,
            hasHands = true,
            bufferFrameCount = 16
        )
        // First frame alone does not switch to hello
        assertEquals(RecognitionState.RECOGNIZING, f1.state)
        assertEquals("IDLE", f1.recognizedSign)
        assertFalse(f1.isStable)

        // Frame 2 with confidence 0.50f (cumulative score = 1.00 >= 0.525, votes = 2)
        val f2 = stabilizer.update(
            predictedClass = "hello",
            predictedClassId = 5,
            confidence = 0.50f,
            hasHands = true,
            bufferFrameCount = 17
        )
        // 2-frame consensus confirms sign
        assertEquals(RecognitionState.ACTIVE_PREDICTION, f2.state)
        assertEquals("hello", f2.recognizedSign)
        assertEquals(5, f2.classId)
        assertEquals(0.50f, f2.confidence, 1e-4f)
        assertTrue(f2.isStable)
    }

    @Test
    fun testActivePredictionSingleHighConfidenceFrame() {
        // Single frame with very high confidence (>= 2x threshold = 0.70f)
        val state = stabilizer.update(
            predictedClass = "water",
            predictedClassId = 17,
            confidence = 0.85f,
            hasHands = true,
            bufferFrameCount = 16
        )
        assertEquals(RecognitionState.ACTIVE_PREDICTION, state.state)
        assertEquals("water", state.recognizedSign)
        assertEquals(17, state.classId)
        assertEquals(0.85f, state.confidence, 1e-4f)
        assertTrue(state.isStable)
    }

    @Test
    fun testAntiFlickerSingleFrameSpikeRejection() {
        // Establish "hello" with 2 frames
        stabilizer.update("hello", 5, 0.60f, hasHands = true, bufferFrameCount = 16)
        val active = stabilizer.update("hello", 5, 0.60f, hasHands = true, bufferFrameCount = 17)
        assertEquals("hello", active.recognizedSign)
        assertEquals(RecognitionState.ACTIVE_PREDICTION, active.state)

        // Single-frame spike of "dance" with confidence 0.45f
        val spike = stabilizer.update("dance", 23, 0.45f, hasHands = true, bufferFrameCount = 18)

        // Anti-flicker hysteresis MUST reject single spike! "hello" remains active!
        assertEquals(RecognitionState.ACTIVE_PREDICTION, spike.state)
        assertEquals("hello", spike.recognizedSign)
        assertEquals(5, spike.classId)

        // If next frame is back to "hello", "hello" continues smoothly
        val recovered = stabilizer.update("hello", 5, 0.55f, hasHands = true, bufferFrameCount = 19)
        assertEquals(RecognitionState.ACTIVE_PREDICTION, recovered.state)
        assertEquals("hello", recovered.recognizedSign)
    }

    @Test
    fun testHysteresisSwitchingToNewSign() {
        // Establish "hello" with 2 frames of 0.50f (cumulative score 1.00f)
        stabilizer.update("hello", 5, 0.50f, hasHands = true, bufferFrameCount = 16)
        stabilizer.update("hello", 5, 0.50f, hasHands = true, bufferFrameCount = 17)
        assertEquals("hello", stabilizer.currentState().recognizedSign)

        // User genuinely changes gesture to "dance" with higher confidence (0.60f)
        // Frame 1 of "dance" (0.60f) -> score 0.60f < "hello" 1.00f -> rejected as spike/candidate
        val switchF1 = stabilizer.update("dance", 23, 0.60f, hasHands = true, bufferFrameCount = 18)
        assertEquals("hello", switchF1.recognizedSign)

        // Frame 2 of "dance" (0.60f) -> score 1.20f > "hello" 1.00f, votes = 2 -> switches to "dance"!
        val switchF2 = stabilizer.update("dance", 23, 0.60f, hasHands = true, bufferFrameCount = 19)
        assertEquals("dance", switchF2.recognizedSign)
        assertEquals(23, switchF2.classId)
        assertEquals(RecognitionState.ACTIVE_PREDICTION, switchF2.state)
        assertTrue(switchF2.isStable)
    }

    @Test
    fun testSwitchingWithEqualConfidenceTakesMajorityWindow() {
        // Establish "hello" with 2 frames of 0.50f
        stabilizer.update("hello", 5, 0.50f, hasHands = true, bufferFrameCount = 16)
        stabilizer.update("hello", 5, 0.50f, hasHands = true, bufferFrameCount = 17)

        // Frame 1 of "pizza" (0.50f) -> score 0.50f < 1.00f
        val f1 = stabilizer.update("pizza", 10, 0.50f, hasHands = true, bufferFrameCount = 18)
        assertEquals("hello", f1.recognizedSign)

        // Frame 2 of "pizza" (0.50f) -> tied (1.00f vs 1.00f) -> incumbent "hello" maintained
        val f2 = stabilizer.update("pizza", 10, 0.50f, hasHands = true, bufferFrameCount = 19)
        assertEquals("hello", f2.recognizedSign)

        // Frame 3 of "pizza" (0.50f) -> pizza takes majority (1.50f vs 1.00f) -> switches to "pizza"!
        val f3 = stabilizer.update("pizza", 10, 0.50f, hasHands = true, bufferFrameCount = 20)
        assertEquals("pizza", f3.recognizedSign)
        assertEquals(10, f3.classId)
        assertEquals(RecognitionState.ACTIVE_PREDICTION, f3.state)
    }

    @Test
    fun testTrackingLostAndRecoveryWithin11Frames() {
        // Establish confirmed sign "thankyou"
        stabilizer.update("thankyou", 14, 0.80f, hasHands = true, bufferFrameCount = 16)
        val confirmed = stabilizer.currentState()
        assertEquals(RecognitionState.ACTIVE_PREDICTION, confirmed.state)
        assertEquals("thankyou", confirmed.recognizedSign)

        // Hands disappear for 5 frames (momentary occlusion)
        for (i in 1..5) {
            val lostState = stabilizer.update(
                prediction = null,
                hasHands = false,
                bufferFrameCount = 16
            )
            assertEquals(RecognitionState.TRACKING_LOST, lostState.state)
            // Crucial: sign is preserved so HUD does not flicker
            assertEquals("thankyou", lostState.recognizedSign)
            assertEquals(14, lostState.classId)
            assertEquals(i, lostState.consecutiveIdleFrames)
            assertFalse(lostState.isStable)
        }

        // Hands reappear on frame 6
        val recovered = stabilizer.update(
            predictedClass = "thankyou",
            predictedClassId = 14,
            confidence = 0.75f,
            hasHands = true,
            bufferFrameCount = 16
        )
        // Successfully restores to ACTIVE_PREDICTION
        assertEquals(RecognitionState.ACTIVE_PREDICTION, recovered.state)
        assertEquals("thankyou", recovered.recognizedSign)
        assertEquals(0, recovered.consecutiveIdleFrames)
        assertTrue(recovered.isStable)
    }

    @Test
    fun testIdleDropoutAfter12Frames() {
        // Establish "pizza"
        stabilizer.update("pizza", 10, 0.85f, hasHands = true, bufferFrameCount = 16)

        // Disappear for 11 frames -> TRACKING_LOST
        for (i in 1..11) {
            val s = stabilizer.update(prediction = null, hasHands = false)
            assertEquals(RecognitionState.TRACKING_LOST, s.state)
            assertEquals("pizza", s.recognizedSign)
            assertEquals(i, s.consecutiveIdleFrames)
        }

        // Frame 12: Hits idleDropoutFrames (12) -> IDLE state, sign cleared
        val idleState = stabilizer.update(prediction = null, hasHands = false)
        assertEquals(RecognitionState.IDLE, idleState.state)
        assertEquals("IDLE", idleState.recognizedSign)
        assertEquals(-1, idleState.classId)
        assertEquals(0.0f, idleState.confidence, 1e-6f)
        assertEquals(12, idleState.consecutiveIdleFrames)
        assertFalse(idleState.isStable)

        // Subsequent idle frame -> WAITING_FOR_HANDS
        val waitingState = stabilizer.update(prediction = null, hasHands = false)
        assertEquals(RecognitionState.WAITING_FOR_HANDS, waitingState.state)
        assertEquals("IDLE", waitingState.recognizedSign)
        assertEquals(13, waitingState.consecutiveIdleFrames)
    }

    @Test
    fun testStalePredictionDecayOnUncertainFrames() {
        // Establish "yellow"
        stabilizer.update("yellow", 18, 0.75f, hasHands = true, bufferFrameCount = 16)
        assertEquals("yellow", stabilizer.currentState().recognizedSign)

        // 5 consecutive frames with low confidence (< 0.35f, uncertain)
        for (i in 1..5) {
            val s = stabilizer.update("yellow", 18, 0.20f, hasHands = true, bufferFrameCount = 16 + i)
            if (i < 5) {
                // Before window is saturated with uncertain votes, sign lingers
                assertEquals(RecognitionState.ACTIVE_PREDICTION, s.state)
            } else {
                // Window full of UNCERTAIN (5 frames): decays to IDLE
                assertEquals(RecognitionState.RECOGNIZING, s.state)
                assertEquals("IDLE", s.recognizedSign)
                assertEquals(-1, s.classId)
                assertEquals(0.0f, s.confidence, 1e-6f)
                assertFalse(s.isStable)
            }
        }
    }

    @Test
    fun testStructuredPredictionResultInterop() {
        val pred = mockPrediction("family", 25, 0.82f)

        val s = stabilizer.update(
            prediction = pred,
            hasHands = true,
            bufferFrameCount = 16
        )

        assertEquals(RecognitionState.ACTIVE_PREDICTION, s.state)
        assertEquals("family", s.recognizedSign)
        assertEquals(25, s.classId)
        assertEquals(0.82f, s.confidence, 1e-4f)
        assertEquals(3, s.top3.size)
        assertEquals("family", s.top3[0].className)
        assertEquals(25, s.top3[0].classId)
        assertEquals(0.82f, s.top3[0].confidence, 1e-4f)
        assertEquals("alt_1", s.top3[1].className)
        assertEquals("alt_2", s.top3[2].className)
        assertTrue(s.isStable)
    }

    @Test
    fun testExplicitReset() {
        stabilizer.update("kitchen", 30, 0.88f, hasHands = true, bufferFrameCount = 16)
        assertEquals("kitchen", stabilizer.currentState().recognizedSign)

        stabilizer.reset()

        val afterReset = stabilizer.currentState()
        assertEquals(RecognitionState.WAITING_FOR_HANDS, afterReset.state)
        assertEquals("IDLE", afterReset.recognizedSign)
        assertEquals(-1, afterReset.classId)
        assertEquals(0.0f, afterReset.confidence, 1e-6f)
        assertTrue(afterReset.top3.isEmpty())
        assertEquals(0, afterReset.consecutiveIdleFrames)
        assertFalse(afterReset.isStable)
    }
}
