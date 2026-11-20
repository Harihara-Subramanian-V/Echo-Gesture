package com.echogesture.app.stabilization

import com.echogesture.app.inference.PredictionCandidate
import com.echogesture.app.inference.PredictionResult
import java.util.ArrayDeque

/**
 * Live Sign Language Recognition Pipeline State Machine for EchoGesture Phase 7C.
 */
enum class RecognitionState {
    /** No hands detected yet; awaiting user interaction. */
    WAITING_FOR_HANDS,

    /** Hands detected; sequence accumulating in buffer (< minFramesForInference). */
    COLLECTING_FRAMES,

    /** Active sequence running through inference model; awaiting stabilized consensus. */
    RECOGNIZING,

    /** Confirmed sign active and displayed to user. */
    ACTIVE_PREDICTION,

    /** Sequence reset / idle timeout reached (>= idleDropoutFrames without hands). */
    IDLE,

    /** Hand disappeared abruptly after active prediction (1 to 11 empty frames). */
    TRACKING_LOST
}

/**
 * Ranked prediction candidate representing a class label, ID, and confidence.
 */
data class ClassPrediction(
    val className: String,
    val classId: Int,
    val confidence: Float
) {
    constructor(candidate: PredictionCandidate) : this(
        className = candidate.className,
        classId = candidate.classId,
        confidence = candidate.confidence
    )
}

/**
 * Raw instantaneous prediction input to [PredictionStabilizer].
 */
data class RawPrediction(
    val className: String,
    val classId: Int,
    val confidence: Float,
    val top3: List<ClassPrediction> = emptyList()
)

/**
 * Immutable snapshot of stabilized prediction state.
 */
data class StabilizedPredictionState(
    val state: RecognitionState,
    val recognizedSign: String,
    val classId: Int,
    val confidence: Float,
    val top3: List<ClassPrediction>,
    val consecutiveIdleFrames: Int,
    val isStable: Boolean
)

/**
 * Temporal hysteresis and weighted voting filter for online predictions.
 *
 * Ports the Python [PredictionStabilizer] from `scripts/live_inference.py` (lines 269-376)
 * to native Android Kotlin, implementing the complete 6-state recognition state machine:
 *
 * 1. [RecognitionState.WAITING_FOR_HANDS]: No hands detected.
 * 2. [RecognitionState.COLLECTING_FRAMES]: Hands detected, buffer accumulating (< minFramesForInference).
 * 3. [RecognitionState.RECOGNIZING]: Active sequence running through inference model.
 * 4. [RecognitionState.ACTIVE_PREDICTION]: Confirmed sign active and displayed.
 * 5. [RecognitionState.IDLE]: Hand absence / sequence reset (>= idleDropoutFrames).
 * 6. [RecognitionState.TRACKING_LOST]: Hands disappeared abruptly after active prediction (1 to 11 empty frames).
 *
 * Hysteresis Rules:
 * - Maintaining an existing sign requires standard threshold (0.35f).
 * - Switching to a new sign requires 2-frame consensus or score >= 2x threshold (0.70f).
 * - Single-frame spikes do NOT switch the stabilized sign.
 * - Stale predictions decay to IDLE when history window fills with UNCERTAIN predictions.
 */
class PredictionStabilizer(
    val windowSize: Int = DEFAULT_WINDOW_SIZE,
    val confidenceThreshold: Float = DEFAULT_CONFIDENCE_THRESHOLD,
    val idleDropoutFrames: Int = DEFAULT_IDLE_DROPOUT_FRAMES,
    val uncertainDropoutFrames: Int = DEFAULT_UNCERTAIN_DROPOUT_FRAMES,
    val minFramesForInference: Int = DEFAULT_MIN_FRAMES_FOR_INFERENCE
) {
    companion object {
        const val DEFAULT_WINDOW_SIZE = 5
        const val DEFAULT_CONFIDENCE_THRESHOLD = 0.35f
        const val DEFAULT_IDLE_DROPOUT_FRAMES = 12
        const val DEFAULT_UNCERTAIN_DROPOUT_FRAMES = 5
        const val DEFAULT_MIN_FRAMES_FOR_INFERENCE = 16

        const val IDLE_SIGN = "IDLE"
        const val UNCERTAIN_SIGN = "UNCERTAIN"
    }

    private data class HistoryItem(
        val className: String,
        val classId: Int,
        val confidence: Float
    )

    private val lock = Any()
    private val history = ArrayDeque<HistoryItem>(windowSize)
    private var consecutiveIdleFrames = 0
    private var consecutiveUncertainFrames = 0

    private var currentStabilizedClass: String = IDLE_SIGN
    private var currentStabilizedId: Int = -1
    private var currentStabilizedConf: Float = 0.0f
    private var currentState: RecognitionState = RecognitionState.WAITING_FOR_HANDS
    private var lastTop3: List<ClassPrediction> = emptyList()

    /**
     * Ingest a raw inference result and hand tracking status, returning the updated stabilized state.
     *
     * @param prediction Structured model prediction result, or null if inference did not run.
     * @param hasHands True if at least one hand is currently detected in the frame.
     * @param bufferFrameCount Current number of frames accumulated in the temporal buffer.
     */
    fun update(
        prediction: PredictionResult?,
        hasHands: Boolean,
        bufferFrameCount: Int = minFramesForInference
    ): StabilizedPredictionState = synchronized(lock) {
        val raw = if (prediction != null) {
            RawPrediction(
                className = prediction.topClassName,
                classId = prediction.topClassId,
                confidence = prediction.topConfidence,
                top3 = prediction.top5.take(3).map { ClassPrediction(it) }
            )
        } else {
            null
        }
        return updateInternal(raw, hasHands, bufferFrameCount)
    }

    /**
     * Convenience overload accepting [RawPrediction].
     */
    fun update(
        rawPrediction: RawPrediction?,
        hasHands: Boolean,
        bufferFrameCount: Int = minFramesForInference
    ): StabilizedPredictionState = synchronized(lock) {
        return updateInternal(rawPrediction, hasHands, bufferFrameCount)
    }

    /**
     * Convenience overload accepting individual prediction values.
     */
    fun update(
        predictedClass: String,
        predictedClassId: Int,
        confidence: Float,
        top3: List<ClassPrediction> = emptyList(),
        hasHands: Boolean = true,
        bufferFrameCount: Int = minFramesForInference
    ): StabilizedPredictionState = synchronized(lock) {
        val raw = RawPrediction(
            className = predictedClass,
            classId = predictedClassId,
            confidence = confidence,
            top3 = top3
        )
        return updateInternal(raw, hasHands, bufferFrameCount)
    }

    private fun updateInternal(
        rawPred: RawPrediction?,
        hasHands: Boolean,
        bufferFrameCount: Int
    ): StabilizedPredictionState {
        // --- 1. Hand Absence Handling ---
        if (!hasHands) {
            consecutiveIdleFrames++
            lastTop3 = emptyList()

            if (consecutiveIdleFrames >= idleDropoutFrames) {
                // Hand absence >= 12 frames: trigger idle reset
                resetInternalState()

                val targetState = if (consecutiveIdleFrames == idleDropoutFrames) {
                    RecognitionState.IDLE
                } else {
                    RecognitionState.WAITING_FOR_HANDS
                }
                currentState = targetState

                return StabilizedPredictionState(
                    state = targetState,
                    recognizedSign = IDLE_SIGN,
                    classId = -1,
                    confidence = 0.0f,
                    top3 = emptyList(),
                    consecutiveIdleFrames = consecutiveIdleFrames,
                    isStable = false
                )
            }

            // Hand absent for 1 <= consecutiveIdleFrames < idleDropoutFrames
            return when (currentState) {
                RecognitionState.ACTIVE_PREDICTION, RecognitionState.TRACKING_LOST -> {
                    currentState = RecognitionState.TRACKING_LOST
                    StabilizedPredictionState(
                        state = RecognitionState.TRACKING_LOST,
                        recognizedSign = currentStabilizedClass,
                        classId = currentStabilizedId,
                        confidence = currentStabilizedConf,
                        top3 = emptyList(),
                        consecutiveIdleFrames = consecutiveIdleFrames,
                        isStable = false
                    )
                }
                RecognitionState.RECOGNIZING -> {
                    StabilizedPredictionState(
                        state = RecognitionState.RECOGNIZING,
                        recognizedSign = IDLE_SIGN,
                        classId = -1,
                        confidence = 0.0f,
                        top3 = emptyList(),
                        consecutiveIdleFrames = consecutiveIdleFrames,
                        isStable = false
                    )
                }
                else -> {
                    currentState = RecognitionState.WAITING_FOR_HANDS
                    StabilizedPredictionState(
                        state = RecognitionState.WAITING_FOR_HANDS,
                        recognizedSign = IDLE_SIGN,
                        classId = -1,
                        confidence = 0.0f,
                        top3 = emptyList(),
                        consecutiveIdleFrames = consecutiveIdleFrames,
                        isStable = false
                    )
                }
            }
        }

        // --- 2. Hands Present ---
        consecutiveIdleFrames = 0

        // Incomplete buffer accumulation (< minFramesForInference)
        if (bufferFrameCount < minFramesForInference) {
            if (currentState == RecognitionState.TRACKING_LOST && currentStabilizedClass != IDLE_SIGN) {
                // Hands reappeared within 11 frames: restore active prediction
                currentState = RecognitionState.ACTIVE_PREDICTION
                return StabilizedPredictionState(
                    state = RecognitionState.ACTIVE_PREDICTION,
                    recognizedSign = currentStabilizedClass,
                    classId = currentStabilizedId,
                    confidence = currentStabilizedConf,
                    top3 = lastTop3,
                    consecutiveIdleFrames = 0,
                    isStable = true
                )
            }
            currentState = RecognitionState.COLLECTING_FRAMES
            return StabilizedPredictionState(
                state = RecognitionState.COLLECTING_FRAMES,
                recognizedSign = IDLE_SIGN,
                classId = -1,
                confidence = 0.0f,
                top3 = emptyList(),
                consecutiveIdleFrames = 0,
                isStable = false
            )
        }

        // Buffer full enough for inference, but no prediction produced for this frame
        if (rawPred == null) {
            if (currentState == RecognitionState.TRACKING_LOST && currentStabilizedClass != IDLE_SIGN) {
                currentState = RecognitionState.ACTIVE_PREDICTION
                return StabilizedPredictionState(
                    state = RecognitionState.ACTIVE_PREDICTION,
                    recognizedSign = currentStabilizedClass,
                    classId = currentStabilizedId,
                    confidence = currentStabilizedConf,
                    top3 = lastTop3,
                    consecutiveIdleFrames = 0,
                    isStable = true
                )
            }
            val state = if (currentStabilizedClass != IDLE_SIGN) {
                RecognitionState.ACTIVE_PREDICTION
            } else {
                RecognitionState.RECOGNIZING
            }
            currentState = state
            val isStable = (state == RecognitionState.ACTIVE_PREDICTION)
            return StabilizedPredictionState(
                state = state,
                recognizedSign = currentStabilizedClass,
                classId = currentStabilizedId,
                confidence = currentStabilizedConf,
                top3 = lastTop3,
                consecutiveIdleFrames = 0,
                isStable = isStable
            )
        }

        // --- 3. Process Raw Prediction through Voting History ---
        lastTop3 = rawPred.top3
        val pClass = rawPred.className
        val pId = rawPred.classId
        val pConf = rawPred.confidence

        // Vote if confidence passes minimum floor
        if (pConf >= confidenceThreshold) {
            if (history.size >= windowSize) {
                history.removeFirst()
            }
            history.addLast(HistoryItem(pClass, pId, pConf))
        } else {
            if (history.size >= windowSize) {
                history.removeFirst()
            }
            history.addLast(HistoryItem(UNCERTAIN_SIGN, -1, pConf))
        }

        // Weighted confidence voting across window
        val classScores = mutableMapOf<Pair<String, Int>, Float>()
        for (item in history) {
            if (item.className == UNCERTAIN_SIGN) continue
            val key = Pair(item.className, item.classId)
            classScores[key] = (classScores[key] ?: 0.0f) + item.confidence
        }

        if (classScores.isEmpty()) {
            consecutiveUncertainFrames++
            // Decay stale predictions once window is full or uncertainDropoutFrames reached
            if (history.size >= windowSize || consecutiveUncertainFrames >= uncertainDropoutFrames) {
                currentStabilizedClass = IDLE_SIGN
                currentStabilizedId = -1
                currentStabilizedConf = 0.0f
            }

            val state = if (currentStabilizedClass != IDLE_SIGN) {
                RecognitionState.ACTIVE_PREDICTION
            } else {
                RecognitionState.RECOGNIZING
            }
            currentState = state
            return StabilizedPredictionState(
                state = state,
                recognizedSign = currentStabilizedClass,
                classId = currentStabilizedId,
                confidence = currentStabilizedConf,
                top3 = lastTop3,
                consecutiveIdleFrames = 0,
                isStable = false
            )
        }

        consecutiveUncertainFrames = 0

        // Find leading candidate
        var bestKey: Pair<String, Int>? = null
        var bestScore = -1.0f
        for ((key, score) in classScores) {
            if (score > bestScore) {
                bestScore = score
                bestKey = key
            }
        }
        val (cName, cId) = bestKey!!
        val votesCount = history.count { it.className == cName }

        // --- 4. Hysteresis Evaluation ---
        val canSwitch: Boolean = if (cName == currentStabilizedClass) {
            // Maintaining existing sign requires standard threshold
            bestScore >= confidenceThreshold
        } else {
            // Switching to a new sign requires 2-frame consensus or score >= 2x threshold (0.70f)
            val meetsMinMass = bestScore >= (confidenceThreshold * 1.5f)
            val meetsConsensusOrStrong = (votesCount >= 2 || bestScore >= (confidenceThreshold * 2.0f))
            meetsMinMass && meetsConsensusOrStrong
        }

        if (canSwitch) {
            currentStabilizedClass = cName
            currentStabilizedId = cId
            val matchingConfs = history.filter { it.className == cName }.map { it.confidence }
            currentStabilizedConf = if (matchingConfs.isNotEmpty()) {
                matchingConfs.average().toFloat()
            } else {
                pConf
            }
        }

        val hasActive = (currentStabilizedClass != IDLE_SIGN)
        val state = if (hasActive) RecognitionState.ACTIVE_PREDICTION else RecognitionState.RECOGNIZING
        val activeVotes = history.count { it.className == currentStabilizedClass }
        val isStable = hasActive && (activeVotes >= 2 || currentStabilizedConf >= (confidenceThreshold * 2.0f))
        currentState = state

        return StabilizedPredictionState(
            state = state,
            recognizedSign = currentStabilizedClass,
            classId = currentStabilizedId,
            confidence = currentStabilizedConf,
            top3 = lastTop3,
            consecutiveIdleFrames = 0,
            isStable = isStable
        )
    }

    /**
     * Complete reset of the stabilizer to initial WAITING_FOR_HANDS state.
     */
    fun reset() = synchronized(lock) {
        resetInternalState()
        consecutiveIdleFrames = 0
        currentState = RecognitionState.WAITING_FOR_HANDS
    }

    private fun resetInternalState() {
        history.clear()
        consecutiveUncertainFrames = 0
        currentStabilizedClass = IDLE_SIGN
        currentStabilizedId = -1
        currentStabilizedConf = 0.0f
        lastTop3 = emptyList()
    }

    /**
     * Inspect current snapshot without altering state.
     */
    fun currentState(): StabilizedPredictionState = synchronized(lock) {
        val hasActive = (currentStabilizedClass != IDLE_SIGN)
        val activeVotes = history.count { it.className == currentStabilizedClass }
        val isStable = hasActive && (activeVotes >= 2 || currentStabilizedConf >= (confidenceThreshold * 2.0f))
        return StabilizedPredictionState(
            state = currentState,
            recognizedSign = currentStabilizedClass,
            classId = currentStabilizedId,
            confidence = currentStabilizedConf,
            top3 = lastTop3,
            consecutiveIdleFrames = consecutiveIdleFrames,
            isStable = isStable
        )
    }
}
