package com.echogesture.app.speech

import com.echogesture.app.engine.PipelineLatencies
import com.echogesture.app.engine.PipelineUiState
import com.echogesture.app.stabilization.RecognitionState
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

/**
 * Exhaustive unit test suite for [PredictionSpeechCoordinator].
 *
 * Verifies all EchoGesture Phase 7D requirements:
 * 1. Duplicate suppression (identical consecutive predictions blocked).
 * 2. Consecutive different signs (cooldown enforcement).
 * 3. Blank, IDLE, and UNCERTAIN filtering.
 * 4. Tracking lost and pipeline transition states.
 * 5. Cooldown window precision and edge cases.
 * 6. Master enable switch and TTS manager enable switch toggles.
 * 7. Reset on IDLE / WAITING_FOR_HANDS recovery.
 * 8. Replay function delegation and error handling.
 * 9. Clear function state reset and audio termination.
 * 10. Rapid prediction fluctuations and fluttering inputs.
 * 11. Confidence threshold enforcement.
 * 12. Non-default policy configurations (suppressDuplicates=false, requireStableState=false).
 */
class PredictionSpeechCoordinatorTest {

    private lateinit var fakeTts: FakeTextToSpeechManager
    private var simulatedTime: Long = 10_000L

    private class FakeTextToSpeechManager : ITextToSpeechManager {
        override var isInitialized: Boolean = true
        override var isSpeaking: Boolean = false
        override var isEnabled: Boolean = true
        override var lastSpokenText: String? = null

        val spokenHistory = mutableListOf<String>()
        var speakCalls = 0
        var replayCalls = 0
        var stopCalls = 0
        var shutdownCalls = 0

        var shouldSucceedSpeak = true
        var shouldSucceedReplay = true

        override fun initialize(onReady: ((Boolean) -> Unit)?) {
            isInitialized = true
            onReady?.invoke(true)
        }

        override fun speak(text: String, queueMode: Int): Boolean {
            speakCalls++
            if (!shouldSucceedSpeak) return false
            lastSpokenText = text
            spokenHistory.add(text)
            isSpeaking = true
            return true
        }

        override fun replay(): Boolean {
            replayCalls++
            if (!shouldSucceedReplay || lastSpokenText == null) return false
            spokenHistory.add(lastSpokenText!!)
            isSpeaking = true
            return true
        }

        override fun stop() {
            stopCalls++
            isSpeaking = false
        }

        override fun shutdown() {
            shutdownCalls++
            isInitialized = false
        }
    }

    private fun createCoordinator(
        config: SpeechPolicyConfig = SpeechPolicyConfig()
    ): PredictionSpeechCoordinator {
        return PredictionSpeechCoordinator(
            ttsManager = fakeTts,
            config = config,
            timeProvider = { simulatedTime }
        )
    }

    private fun createUiState(
        sign: String = "hello",
        state: RecognitionState = RecognitionState.ACTIVE_PREDICTION,
        confidence: Float = 0.85f
    ): PipelineUiState {
        return PipelineUiState(
            state = state,
            recognizedSign = sign,
            confidence = confidence,
            topCandidates = emptyList(),
            detectedHands = emptyList(),
            latencies = PipelineLatencies(),
            fps = 30.0,
            isFrontCamera = true,
            consecutiveIdleFrames = 0,
            bufferFill = 16
        )
    }

    @Before
    fun setUp() {
        fakeTts = FakeTextToSpeechManager()
        simulatedTime = 10_000L
    }

    @Test
    fun testInitialState() {
        val coordinator = createCoordinator()
        assertTrue(coordinator.isEnabled)
        assertNull(coordinator.lastSpokenSign)
        assertEquals(0L, coordinator.lastSpokenTimestamp)
    }

    @Test
    fun testValidPredictionTriggersSpeech() {
        val coordinator = createCoordinator()
        val state = createUiState(sign = "HELLO", confidence = 0.90f)

        val triggered = coordinator.onPredictionState(state)

        assertTrue(triggered)
        assertEquals(1, fakeTts.speakCalls)
        assertEquals("HELLO", fakeTts.lastSpokenText)
        assertEquals("HELLO", coordinator.lastSpokenSign)
        assertEquals(10_000L, coordinator.lastSpokenTimestamp)
        assertEquals(listOf("HELLO"), fakeTts.spokenHistory)
    }

    @Test
    fun testRepeatedIdenticalPredictionsAreSuppressed() {
        val coordinator = createCoordinator()
        val state1 = createUiState(sign = "HELLO")

        assertTrue(coordinator.onPredictionState(state1))
        assertEquals(1, fakeTts.speakCalls)

        // Advance time well past cooldown (2000 ms)
        simulatedTime += 2000L

        // Same sign emitted again
        val state2 = createUiState(sign = "HELLO")
        val repeated = coordinator.onPredictionState(state2)

        assertFalse("Duplicate sign must be suppressed", repeated)
        assertEquals(1, fakeTts.speakCalls)
        assertEquals("HELLO", coordinator.lastSpokenSign)
    }

    @Test
    fun testDuplicateSuppressionIsCaseInsensitive() {
        val coordinator = createCoordinator()
        assertTrue(coordinator.onPredictionState(createUiState(sign = "ThankYou")))
        assertEquals(1, fakeTts.speakCalls)

        simulatedTime += 3000L

        assertFalse(coordinator.onPredictionState(createUiState(sign = "THANKYOU")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "thankyou")))
        assertEquals(1, fakeTts.speakCalls)
    }

    @Test
    fun testConsecutiveDifferentSignsRespectCooldown() {
        val coordinator = createCoordinator(SpeechPolicyConfig(cooldownMs = 1500L))

        // Sign 1: HELLO at 10_000ms
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(listOf("HELLO"), fakeTts.spokenHistory)

        // Sign 2: YES at 10_500ms (500ms elapsed < 1500ms cooldown) -> blocked
        simulatedTime = 10_500L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "YES")))
        assertEquals(listOf("HELLO"), fakeTts.spokenHistory)

        // Sign 2: YES at 11_499ms (1499ms elapsed < 1500ms cooldown) -> blocked
        simulatedTime = 11_499L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "YES")))
        assertEquals(listOf("HELLO"), fakeTts.spokenHistory)

        // Sign 2: YES at 11_500ms (1500ms elapsed >= 1500ms cooldown) -> spoken
        simulatedTime = 11_500L
        assertTrue(coordinator.onPredictionState(createUiState(sign = "YES")))
        assertEquals(listOf("HELLO", "YES"), fakeTts.spokenHistory)
        assertEquals("YES", coordinator.lastSpokenSign)
        assertEquals(11_500L, coordinator.lastSpokenTimestamp)
    }

    @Test
    fun testBlankIdleAndUncertainPredictionsNeverSpeak() {
        val coordinator = createCoordinator()

        // Blank and whitespace
        assertFalse(coordinator.onPredictionState(createUiState(sign = "")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "   ")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "\t\n")))

        // IDLE in various casings
        assertFalse(coordinator.onPredictionState(createUiState(sign = "IDLE")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "idle")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "Idle")))

        // UNCERTAIN in various casings
        assertFalse(coordinator.onPredictionState(createUiState(sign = "UNCERTAIN")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "uncertain")))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "Uncertain")))

        assertEquals(0, fakeTts.speakCalls)
        assertNull(coordinator.lastSpokenSign)
    }

    @Test
    fun testNonActivePredictionStatesNeverSpeak() {
        val coordinator = createCoordinator(SpeechPolicyConfig(requireStableState = true))

        val nonActiveStates = listOf(
            RecognitionState.WAITING_FOR_HANDS,
            RecognitionState.COLLECTING_FRAMES,
            RecognitionState.RECOGNIZING,
            RecognitionState.TRACKING_LOST,
            RecognitionState.IDLE
        )

        for (st in nonActiveStates) {
            val state = createUiState(sign = "HELLO", state = st, confidence = 0.95f)
            assertFalse("State $st should not trigger speech", coordinator.onPredictionState(state))
        }

        assertEquals(0, fakeTts.speakCalls)
    }

    @Test
    fun testTrackingLostAndRecoveryFlow() {
        val coordinator = createCoordinator()

        // 1. User signs "HELLO" in ACTIVE_PREDICTION
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(listOf("HELLO"), fakeTts.spokenHistory)

        // 2. User momentarily drops hands -> TRACKING_LOST
        simulatedTime += 200L
        assertFalse(coordinator.onPredictionState(
            createUiState(sign = "HELLO", state = RecognitionState.TRACKING_LOST)
        ))

        // 3. User recovers within tracking lost window -> still "HELLO", suppressed duplicate
        simulatedTime += 200L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO")))

        // 4. User transitions to new sign "PLEASE" after cooldown
        simulatedTime += 1500L
        assertTrue(coordinator.onPredictionState(createUiState(sign = "PLEASE")))
        assertEquals(listOf("HELLO", "PLEASE"), fakeTts.spokenHistory)
    }

    @Test
    fun testConfidenceThresholdEnforcement() {
        val coordinator = createCoordinator(SpeechPolicyConfig(minConfidence = 0.50f))

        // Below threshold
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO", confidence = 0.499f)))
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO", confidence = 0.0f)))
        assertEquals(0, fakeTts.speakCalls)

        // At or above threshold
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO", confidence = 0.50f)))
        assertEquals(1, fakeTts.speakCalls)

        // Custom high threshold
        val strictCoordinator = createCoordinator(SpeechPolicyConfig(minConfidence = 0.80f))
        assertFalse(strictCoordinator.onPredictionState(createUiState(sign = "HELLO", confidence = 0.79f)))
        assertTrue(strictCoordinator.onPredictionState(createUiState(sign = "HELLO", confidence = 0.81f)))
    }

    @Test
    fun testSpeechDisabledToggles() {
        val coordinator = createCoordinator()

        // Disable coordinator master switch
        coordinator.isEnabled = false
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(0, fakeTts.speakCalls)

        // Re-enable coordinator
        coordinator.isEnabled = true
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(1, fakeTts.speakCalls)

        // Disable TTS manager switch
        fakeTts.isEnabled = false
        simulatedTime += 2000L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "WORLD")))
        assertEquals(1, fakeTts.speakCalls)

        // Re-enable TTS manager switch
        fakeTts.isEnabled = true
        assertTrue(coordinator.onPredictionState(createUiState(sign = "WORLD")))
        assertEquals(2, fakeTts.speakCalls)
    }

    @Test
    fun testResetOnIdleAllowsSameSignRepetition() {
        val coordinator = createCoordinator(SpeechPolicyConfig(cooldownMs = 1500L, resetOnIdle = true))

        // 1. Sign "HELLO"
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals("HELLO", coordinator.lastSpokenSign)

        // 2. Same sign is suppressed while continuously active
        simulatedTime += 2000L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO")))

        // 3. User drops hands and pipeline transitions to IDLE
        simulatedTime += 500L
        val idleState = createUiState(sign = "IDLE", state = RecognitionState.IDLE, confidence = 0.0f)
        assertFalse(coordinator.onPredictionState(idleState))
        assertNull("lastSpokenSign must be reset on IDLE", coordinator.lastSpokenSign)

        // 4. User performs "HELLO" again after idle break (past cooldown)
        simulatedTime += 1500L
        assertTrue("Same sign must speak again after idle reset", coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(listOf("HELLO", "HELLO"), fakeTts.spokenHistory)
    }

    @Test
    fun testResetOnWaitingForHandsAllowsSameSignRepetition() {
        val coordinator = createCoordinator(SpeechPolicyConfig(cooldownMs = 1500L, resetOnIdle = true))

        // Sign "YES"
        assertTrue(coordinator.onPredictionState(createUiState(sign = "YES")))

        // WAITING_FOR_HANDS resets duplicate history
        simulatedTime += 2000L
        val waitingState = createUiState(sign = "IDLE", state = RecognitionState.WAITING_FOR_HANDS, confidence = 0.0f)
        assertFalse(coordinator.onPredictionState(waitingState))
        assertNull(coordinator.lastSpokenSign)

        // Sign "YES" again speaks
        assertTrue(coordinator.onPredictionState(createUiState(sign = "YES")))
        assertEquals(listOf("YES", "YES"), fakeTts.spokenHistory)
    }

    @Test
    fun testResetOnIdleDisabledPreservesDuplicateSuppression() {
        val coordinator = createCoordinator(SpeechPolicyConfig(cooldownMs = 1500L, resetOnIdle = false))

        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals("HELLO", coordinator.lastSpokenSign)

        simulatedTime += 2000L
        coordinator.onPredictionState(createUiState(sign = "IDLE", state = RecognitionState.IDLE))

        // Duplicate still suppressed because resetOnIdle is false
        assertEquals("HELLO", coordinator.lastSpokenSign)
        assertFalse(coordinator.onPredictionState(createUiState(sign = "HELLO")))
    }

    @Test
    fun testReplayFunction() {
        val coordinator = createCoordinator()

        // Replay before anything was spoken returns false
        assertFalse(coordinator.replay())
        assertEquals(0, fakeTts.replayCalls)

        // Speak a sign
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(1, fakeTts.speakCalls)

        // Replay delegates to TTS manager replay
        assertTrue(coordinator.replay())
        assertEquals(1, fakeTts.replayCalls)
        assertEquals(listOf("HELLO", "HELLO"), fakeTts.spokenHistory)

        // Replay when disabled returns false
        coordinator.isEnabled = false
        assertFalse(coordinator.replay())

        coordinator.isEnabled = true
        fakeTts.isEnabled = false
        assertFalse(coordinator.replay())
    }

    @Test
    fun testReplayFallbackToSpeakWhenTtsReplayFails() {
        val coordinator = createCoordinator()
        fakeTts.shouldSucceedReplay = false

        assertTrue(coordinator.onPredictionState(createUiState(sign = "FALLBACK")))
        assertEquals(1, fakeTts.speakCalls)

        // ttsManager.replay() returns false, coordinator falls back to ttsManager.speak(lastSpokenSign)
        assertTrue(coordinator.replay())
        assertEquals(1, fakeTts.replayCalls)
        assertEquals(2, fakeTts.speakCalls)
        assertEquals(listOf("FALLBACK", "FALLBACK"), fakeTts.spokenHistory)
    }

    @Test
    fun testClearFunction() {
        val coordinator = createCoordinator()

        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals("HELLO", coordinator.lastSpokenSign)
        assertEquals(10_000L, coordinator.lastSpokenTimestamp)

        // Call clear
        coordinator.clear()
        assertNull(coordinator.lastSpokenSign)
        assertEquals(0L, coordinator.lastSpokenTimestamp)
        assertEquals(1, fakeTts.stopCalls)

        // Immediately signing "HELLO" should speak because state is fully cleared
        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        assertEquals(listOf("HELLO", "HELLO"), fakeTts.spokenHistory)
    }

    @Test
    fun testStopFunction() {
        val coordinator = createCoordinator()

        assertTrue(coordinator.onPredictionState(createUiState(sign = "HELLO")))
        coordinator.stop()

        assertEquals(1, fakeTts.stopCalls)
        assertEquals("HELLO", coordinator.lastSpokenSign)
        assertEquals(10_000L, coordinator.lastSpokenTimestamp)
    }

    @Test
    fun testRapidPredictionFluctuations() {
        val coordinator = createCoordinator(SpeechPolicyConfig(cooldownMs = 1500L, minConfidence = 0.50f))

        // t = 10_000ms: Sign "A" recognized with high confidence -> SPOKEN
        assertTrue(coordinator.onPredictionState(createUiState(sign = "A", confidence = 0.90f)))

        // t = 10_033ms: Model flickers to "B" (33ms later) -> BLOCKED by cooldown
        simulatedTime += 33L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "B", confidence = 0.92f)))

        // t = 10_066ms: Model flickers to "C" (66ms later) -> BLOCKED by cooldown
        simulatedTime += 33L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "C", confidence = 0.88f)))

        // t = 10_100ms: State drops to RECOGNIZING -> BLOCKED by state requirement
        simulatedTime += 34L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "D", state = RecognitionState.RECOGNIZING)))

        // t = 11_501ms: Stable prediction "D" arrives past cooldown (1501ms elapsed) -> SPOKEN
        simulatedTime = 11_501L
        assertTrue(coordinator.onPredictionState(createUiState(sign = "D", confidence = 0.95f)))

        assertEquals(listOf("A", "D"), fakeTts.spokenHistory)
    }

    @Test
    fun testDisabledDuplicateSuppressionAllowsImmediateSameSignAfterCooldown() {
        val coordinator = createCoordinator(
            SpeechPolicyConfig(cooldownMs = 1000L, suppressDuplicates = false)
        )

        assertTrue(coordinator.onPredictionState(createUiState(sign = "REPEAT")))

        // Before cooldown -> blocked
        simulatedTime += 500L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "REPEAT")))

        // After cooldown -> allowed because duplicate suppression is false
        simulatedTime += 600L
        assertTrue(coordinator.onPredictionState(createUiState(sign = "REPEAT")))
        assertEquals(listOf("REPEAT", "REPEAT"), fakeTts.spokenHistory)
    }

    @Test
    fun testDisabledRequireStableStateAllowsNonActiveStates() {
        val coordinator = createCoordinator(
            SpeechPolicyConfig(cooldownMs = 1000L, requireStableState = false)
        )

        assertTrue(coordinator.onPredictionState(
            createUiState(sign = "EARLY", state = RecognitionState.RECOGNIZING, confidence = 0.80f)
        ))
        assertEquals(listOf("EARLY"), fakeTts.spokenHistory)
    }

    @Test
    fun testCooldownZeroAllowsImmediateDifferentSigns() {
        val coordinator = createCoordinator(
            SpeechPolicyConfig(cooldownMs = 0L)
        )

        assertTrue(coordinator.onPredictionState(createUiState(sign = "FIRST")))
        // Same millisecond, different sign
        assertTrue(coordinator.onPredictionState(createUiState(sign = "SECOND")))
        assertEquals(listOf("FIRST", "SECOND"), fakeTts.spokenHistory)
    }

    @Test
    fun testFirstSignNotBlockedWhenTimeStartsAtZero() {
        var zeroTime = 0L
        val coordinator = PredictionSpeechCoordinator(
            ttsManager = fakeTts,
            config = SpeechPolicyConfig(cooldownMs = 1500L),
            timeProvider = { zeroTime }
        )

        // First sign at time = 0 must speak immediately
        assertTrue(coordinator.onPredictionState(createUiState(sign = "ZERO")))
        assertEquals(listOf("ZERO"), fakeTts.spokenHistory)
        assertEquals(0L, coordinator.lastSpokenTimestamp)

        // Subsequent sign at time = 500 must be blocked by cooldown
        zeroTime = 500L
        assertFalse(coordinator.onPredictionState(createUiState(sign = "ONE")))

        // Subsequent sign at time = 1500 must speak
        zeroTime = 1500L
        assertTrue(coordinator.onPredictionState(createUiState(sign = "ONE")))
        assertEquals(listOf("ZERO", "ONE"), fakeTts.spokenHistory)
    }
}
