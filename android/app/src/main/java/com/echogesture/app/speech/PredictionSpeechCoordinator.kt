package com.echogesture.app.speech

import com.echogesture.app.engine.PipelineUiState
import com.echogesture.app.stabilization.RecognitionState

/**
 * Configuration policy controlling when sign predictions are announced via Text-to-Speech.
 *
 * @param cooldownMs Minimum interval in milliseconds required between consecutive speech announcements.
 * @param minConfidence Minimum prediction confidence threshold [0.0, 1.0] required to trigger speech.
 * @param requireStableState Whether speech requires [RecognitionState.ACTIVE_PREDICTION].
 * @param suppressDuplicates Whether identical consecutive signs are suppressed.
 * @param resetOnIdle Whether transition to [RecognitionState.IDLE] or [RecognitionState.WAITING_FOR_HANDS]
 *                    clears duplicate suppression history.
 */
data class SpeechPolicyConfig(
    val cooldownMs: Long = 1500L,
    val minConfidence: Float = 0.50f,
    val requireStableState: Boolean = true,
    val suppressDuplicates: Boolean = true,
    val resetOnIdle: Boolean = true
)

/**
 * Coordinates and filters sign predictions from the real-time recognition pipeline
 * before dispatching them to the Text-to-Speech (TTS) engine.
 *
 * Enforces cooldown gating, duplicate suppression, state stability, and confidence thresholds.
 */
class PredictionSpeechCoordinator(
    val ttsManager: ITextToSpeechManager,
    val config: SpeechPolicyConfig = SpeechPolicyConfig(),
    val timeProvider: () -> Long = { System.currentTimeMillis() }
) {
    @Volatile
    var isEnabled: Boolean = true

    @Volatile
    var lastSpokenSign: String? = null
        private set

    @Volatile
    var lastSpokenTimestamp: Long = 0L
        private set

    /**
     * Evaluates incoming [PipelineUiState] from the recognition pipeline to determine whether
     * to announce the recognized sign.
     *
     * Invariants:
     * 1. If [isEnabled] is false or [ITextToSpeechManager.isEnabled] is false, returns false.
     * 2. If sign is blank, or equals "IDLE", or equals "UNCERTAIN" (case-insensitive), returns false.
     * 3. If [SpeechPolicyConfig.resetOnIdle] is true and state is IDLE or WAITING_FOR_HANDS,
     *    resets [lastSpokenSign] to null.
     * 4. If [SpeechPolicyConfig.requireStableState] is true and state is not ACTIVE_PREDICTION,
     *    returns false.
     * 5. If confidence is below [SpeechPolicyConfig.minConfidence], returns false.
     * 6. If [SpeechPolicyConfig.suppressDuplicates] is true and sign matches [lastSpokenSign] (case-insensitive),
     *    returns false.
     * 7. If time elapsed since last spoken sign is less than [SpeechPolicyConfig.cooldownMs], returns false.
     * 8. When all criteria pass, invokes [ITextToSpeechManager.speak], updates [lastSpokenSign]
     *    and [lastSpokenTimestamp], and returns true.
     *
     * @param uiState Current pipeline UI state.
     * @return `true` if speech was triggered, `false` otherwise.
     */
    @Synchronized
    fun onPredictionState(uiState: PipelineUiState): Boolean {
        // 3. Reset duplicate tracking on idle or waiting states
        if (config.resetOnIdle && (uiState.state == RecognitionState.IDLE || uiState.state == RecognitionState.WAITING_FOR_HANDS)) {
            lastSpokenSign = null
        }

        // 1. Check master enable switch and TTS manager enable switch
        if (!isEnabled || !ttsManager.isEnabled) {
            return false
        }

        // 2. Reject blank, IDLE, or UNCERTAIN signs
        val sign = uiState.recognizedSign.trim()
        if (sign.isEmpty() || sign.equals("IDLE", ignoreCase = true) || sign.equals("UNCERTAIN", ignoreCase = true)) {
            return false
        }

        // 4. Check stable state requirement
        if (config.requireStableState && uiState.state != RecognitionState.ACTIVE_PREDICTION) {
            return false
        }

        // 5. Check minimum confidence threshold
        if (uiState.confidence < config.minConfidence) {
            return false
        }

        // 6. Duplicate suppression check
        if (config.suppressDuplicates && sign.equals(lastSpokenSign, ignoreCase = true)) {
            return false
        }

        // 7. Cooldown window enforcement
        val now = timeProvider()
        if ((lastSpokenTimestamp > 0L || lastSpokenSign != null) && (now - lastSpokenTimestamp < config.cooldownMs)) {
            return false
        }

        // 8. Announce sign, record history and timestamp
        ttsManager.speak(uiState.recognizedSign)
        lastSpokenSign = uiState.recognizedSign
        lastSpokenTimestamp = now
        return true
    }

    /**
     * Replays the last spoken sign, or delegates to [ITextToSpeechManager.replay].
     *
     * @return `true` if replay was triggered, `false` otherwise.
     */
    @Synchronized
    fun replay(): Boolean {
        if (!isEnabled || !ttsManager.isEnabled) {
            return false
        }
        val sign = lastSpokenSign ?: return false
        if (ttsManager.replay()) {
            return true
        }
        return ttsManager.speak(sign)
    }

    /**
     * Resets spoken state history and stops any active speech playback.
     */
    @Synchronized
    fun clear() {
        lastSpokenSign = null
        lastSpokenTimestamp = 0L
        ttsManager.stop()
    }

    /**
     * Stops any ongoing speech playback without clearing spoken state history.
     */
    fun stop() {
        ttsManager.stop()
    }
}
