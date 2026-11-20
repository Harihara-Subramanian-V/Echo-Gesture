package com.echogesture.app.engine

import android.content.Context
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import com.echogesture.app.feature.DetectedHandCandidate
import com.echogesture.app.feature.FeatureExtractor134D
import com.echogesture.app.feature.HandSlotResolver
import com.echogesture.app.inference.EchoGestureInferenceEngine
import com.echogesture.app.inference.LiteRtModelRunner
import com.echogesture.app.preprocessing.SequenceNormalizer
import com.echogesture.app.preprocessing.SequencePreprocessor263D
import com.echogesture.app.stabilization.ClassPrediction
import com.echogesture.app.stabilization.PredictionStabilizer
import com.echogesture.app.stabilization.RawPrediction
import com.echogesture.app.stabilization.RecognitionState
import com.echogesture.app.stabilization.StabilizedPredictionState
import com.echogesture.app.temporal.LiveTemporalBuffer
import com.echogesture.app.vision.HandLandmarkDetector
import com.echogesture.app.vision.IHandLandmarkDetector
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Breakdown of latency stages (all values in milliseconds).
 */
data class PipelineLatencies(
    val detectionMs: Double = 0.0,
    val featureMs: Double = 0.0,
    val preprocessMs: Double = 0.0,
    val inferenceMs: Double = 0.0,
    val stabilizationMs: Double = 0.0,
    val totalMs: Double = 0.0
)

/**
 * Immutable UI state dispatched to the HUD after each processed frame.
 */
data class PipelineUiState(
    val state: RecognitionState,
    val recognizedSign: String,
    val confidence: Float,
    val topCandidates: List<ClassPrediction>,
    val detectedHands: List<DetectedHandCandidate>,
    val latencies: PipelineLatencies,
    val fps: Double,
    val isFrontCamera: Boolean,
    val consecutiveIdleFrames: Int,
    val bufferFill: Int
)

fun interface PipelineStateListener {
    fun onStateUpdated(uiState: PipelineUiState)
}

/**
 * End-to-End Orchestrator for the EchoGesture real-time sign recognition loop.
 *
 * Coordinates:
 * Camera Frame -> MediaPipe Detection -> Hand Slot Resolution -> 134-D Feature Extraction
 * -> Rolling Buffer -> 263-D Preprocessing -> INT8 LiteRT Inference -> Prediction Stabilization
 * -> UI Telemetry Callback.
 */
class RecognitionPipeline(
    private val context: Context? = null,
    val detector: IHandLandmarkDetector,
    val modelRunner: LiteRtModelRunner,
    val preprocessor: SequencePreprocessor263D,
    val inferenceEngine: EchoGestureInferenceEngine = EchoGestureInferenceEngine(modelRunner, preprocessor),
    val slotResolver: HandSlotResolver = HandSlotResolver(),
    val featureExtractor: FeatureExtractor134D = FeatureExtractor134D(),
    val temporalBuffer: LiveTemporalBuffer = LiveTemporalBuffer(),
    val stabilizer: PredictionStabilizer = PredictionStabilizer(),
    var stateListener: PipelineStateListener? = null
) : AutoCloseable {

    companion object {
        private const val TAG = "RecognitionPipeline"
        private const val FPS_WINDOW = 30
    }

    private val isProcessing = AtomicBoolean(false)
    private var isClosed = false

    // FPS calculation tracking
    private val frameTimestamps = LongArray(FPS_WINDOW)
    private var frameIndex = 0
    private var currentFps = 0.0

    /**
     * Ingests a new camera frame from CameraX.
     * Executes non-blocking; safely drops frames if a prior frame is still being processed.
     */
    fun processFrame(imageProxy: ImageProxy, isFrontCamera: Boolean) {
        if (isClosed) return
        if (!isProcessing.compareAndSet(false, true)) {
            // Drop frame to preserve real-time low latency
            return
        }

        try {
            val tStartTotal = System.nanoTime()

            // Update FPS
            val nowMs = SystemClock.elapsedRealtime()
            frameTimestamps[frameIndex % FPS_WINDOW] = nowMs
            frameIndex++
            if (frameIndex >= FPS_WINDOW) {
                val oldestMs = frameTimestamps[frameIndex % FPS_WINDOW]
                val span = (nowMs - oldestMs) / 1000.0
                if (span > 0.0) {
                    currentFps = (FPS_WINDOW - 1) / span
                }
            }

            val imgW = imageProxy.width
            val imgH = imageProxy.height

            // 1. MediaPipe Hand Landmark Detection
            val t0 = System.nanoTime()
            val detectedHands = detector.detect(imageProxy)
            val t1 = System.nanoTime()
            val detectionMs = (t1 - t0) / 1_000_000.0

            // 2. Hand Slot Resolution & 134-D Feature Extraction
            val resolvedSlots = slotResolver.resolve(detectedHands)
            val frame134 = featureExtractor.extract(resolvedSlots, imgW, imgH)
            val t2 = System.nanoTime()
            val featureMs = (t2 - t1) / 1_000_000.0

            // 3. Temporal Buffer update
            val hasHands = detectedHands.isNotEmpty()
            temporalBuffer.addFrame(frame134, hasHands)
            val bufferCount = temporalBuffer.size

            var preprocessMs = 0.0
            var inferenceMs = 0.0
            var stabilizationMs = 0.0
            var stabilizedState: StabilizedPredictionState

            // 4. Inference & Stabilization Execution
            if (hasHands && bufferCount >= stabilizer.minFramesForInference) {
                val tPre0 = System.nanoTime()
                val sequenceSnapshot = temporalBuffer.getResampledSnapshot(64)
                val tPre1 = System.nanoTime()
                preprocessMs = (tPre1 - tPre0) / 1_000_000.0

                // LiteRT INT8 Model Inference
                val tInf0 = System.nanoTime()
                val predictionResult = inferenceEngine.predict(sequenceSnapshot)
                val tInf1 = System.nanoTime()
                inferenceMs = (tInf1 - tInf0) / 1_000_000.0

                // Stabilization
                val tStab0 = System.nanoTime()
                val rawPred = RawPrediction(
                    className = predictionResult.topClassName,
                    classId = predictionResult.topClassId,
                    confidence = predictionResult.topConfidence,
                    top3 = predictionResult.top5.take(3).map { ClassPrediction(it) }
                )
                stabilizedState = stabilizer.update(rawPred, hasHands, bufferCount)
                val tStab1 = System.nanoTime()
                stabilizationMs = (tStab1 - tStab0) / 1_000_000.0
            } else {
                val tStab0 = System.nanoTime()
                stabilizedState = stabilizer.update(null as RawPrediction?, hasHands, bufferCount)
                val tStab1 = System.nanoTime()
                stabilizationMs = (tStab1 - tStab0) / 1_000_000.0
            }

            val tEndTotal = System.nanoTime()
            val totalMs = (tEndTotal - tStartTotal) / 1_000_000.0

            val latencies = PipelineLatencies(
                detectionMs = detectionMs,
                featureMs = featureMs,
                preprocessMs = preprocessMs,
                inferenceMs = inferenceMs,
                stabilizationMs = stabilizationMs,
                totalMs = totalMs
            )

            val uiState = PipelineUiState(
                state = stabilizedState.state,
                recognizedSign = stabilizedState.recognizedSign,
                confidence = stabilizedState.confidence,
                topCandidates = stabilizedState.top3,
                detectedHands = detectedHands,
                latencies = latencies,
                fps = currentFps,
                isFrontCamera = isFrontCamera,
                consecutiveIdleFrames = stabilizedState.consecutiveIdleFrames,
                bufferFill = bufferCount
            )

            if (frameIndex % 15 == 0) {
                Log.i(
                    TAG,
                    "PIPELINE_TELEMETRY: state=${uiState.state} sign=${uiState.recognizedSign} " +
                    "conf=${String.format("%.3f", uiState.confidence)} hands=${uiState.detectedHands.size} " +
                    "fps=${String.format("%.1f", uiState.fps)} totalMs=${String.format("%.2f", latencies.totalMs)} " +
                    "detectMs=${String.format("%.2f", latencies.detectionMs)} featMs=${String.format("%.2f", latencies.featureMs)} " +
                    "infMs=${String.format("%.2f", latencies.inferenceMs)} stabMs=${String.format("%.2f", latencies.stabilizationMs)}"
                )
            }

            stateListener?.onStateUpdated(uiState)

        } catch (e: Exception) {
            Log.e(TAG, "Error in recognition pipeline frame processing: ${e.message}", e)
        } finally {
            isProcessing.set(false)
        }
    }

    fun reset() {
        temporalBuffer.clear()
        stabilizer.reset()
        slotResolver.reset()
    }

    override fun close() {
        if (!isClosed) {
            isClosed = true
            reset()
            detector.close()
            inferenceEngine.close()
            modelRunner.close()
        }
    }
}
