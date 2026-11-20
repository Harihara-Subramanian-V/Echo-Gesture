package com.echogesture.app.vision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.ImageProxy
import com.echogesture.app.feature.DetectedHandCandidate
import com.echogesture.app.feature.HandSlotResolver
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MediaImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Internal delegate interface wrapping the MediaPipe HandLandmarker API.
 * Facilitates unit testing on host desktop JVMs where native JNI binaries are not available.
 */
internal interface HandLandmarkerEngine : AutoCloseable {
    val isAvailable: Boolean
    val lastConversionMs: Double get() = 0.0
    val lastRawDetectMs: Double get() = 0.0
    fun detect(bitmap: Bitmap, rotationDegrees: Int): HandLandmarkerResult?
    fun detect(imageProxy: ImageProxy): HandLandmarkerResult?
}

/**
 * Production implementation invoking Google MediaPipe Tasks Vision on Android devices.
 */
internal class NativeHandLandmarkerEngine(
    private val landmarker: HandLandmarker
) : HandLandmarkerEngine {

    override val isAvailable: Boolean = true
    override var lastConversionMs: Double = 0.0
        private set
    override var lastRawDetectMs: Double = 0.0
        private set

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): HandLandmarkerResult? {
        val tConv0 = System.nanoTime()
        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()
        val options = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()
        val tConv1 = System.nanoTime()
        lastConversionMs = (tConv1 - tConv0) / 1_000_000.0

        val tDetect0 = System.nanoTime()
        val result = landmarker.detect(mpImage, options)
        val tDetect1 = System.nanoTime()
        lastRawDetectMs = (tDetect1 - tDetect0) / 1_000_000.0

        return result
    }

    override fun detect(imageProxy: ImageProxy): HandLandmarkerResult? {
        val tConv0 = System.nanoTime()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val options = ImageProcessingOptions.builder()
            .setRotationDegrees(rotationDegrees)
            .build()
        val bitmap = imageProxy.toBitmap()
        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()
        val tConv1 = System.nanoTime()
        lastConversionMs = (tConv1 - tConv0) / 1_000_000.0

        val tDetect0 = System.nanoTime()
        val result = landmarker.detect(mpImage, options)
        val tDetect1 = System.nanoTime()
        lastRawDetectMs = (tDetect1 - tDetect0) / 1_000_000.0

        return result
    }

    override fun close() {
        landmarker.close()
    }
}

/**
 * Fallback implementation used during host JVM unit testing or when native MediaPipe JNI is absent.
 */
internal class HostJvmFallbackEngine(
    private var testResultProvider: ((Bitmap?, ImageProxy?) -> HandLandmarkerResult?)? = null
) : HandLandmarkerEngine {

    private var _isClosed = false
    override val isAvailable: Boolean get() = !_isClosed
    override var lastConversionMs: Double = 0.0
    override var lastRawDetectMs: Double = 0.0

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): HandLandmarkerResult? {
        check(!_isClosed) { "HostJvmFallbackEngine is closed" }
        val t0 = System.nanoTime()
        val result = testResultProvider?.invoke(bitmap, null)
        val t1 = System.nanoTime()
        lastConversionMs = 0.0
        lastRawDetectMs = (t1 - t0) / 1_000_000.0
        return result
    }

    override fun detect(imageProxy: ImageProxy): HandLandmarkerResult? {
        check(!_isClosed) { "HostJvmFallbackEngine is closed" }
        val t0 = System.nanoTime()
        val result = testResultProvider?.invoke(null, imageProxy)
        val t1 = System.nanoTime()
        lastConversionMs = 0.0
        lastRawDetectMs = (t1 - t0) / 1_000_000.0
        return result
    }

    fun setTestResult(result: HandLandmarkerResult?) {
        testResultProvider = { _, _ -> result }
    }

    override fun close() {
        _isClosed = true
    }
}

/**
 * Google MediaPipe Tasks Vision Hand Landmarker wrapper for EchoGesture Phase 7C.
 *
 * Implements real-time synchronous hand skeleton landmark extraction for the live vision pipeline:
 * - Model: hand_landmarker.task (Asset)
 * - Number of Hands: Up to 2
 * - Confidence Thresholds: 0.5f (detection, presence, tracking)
 * - Synchronous Image Mode for per-frame processing
 * - Formats landmarks into 63-D float arrays matching [DetectedHandCandidate] requirements
 * - Implements AutoCloseable for deterministic resource deallocation
 */
class HandLandmarkDetector internal constructor(
    private val context: Context?,
    val modelAssetPath: String = DEFAULT_MODEL_ASSET_PATH,
    val numHands: Int = DEFAULT_NUM_HANDS,
    val minHandDetectionConfidence: Float = DEFAULT_DETECTION_CONFIDENCE,
    val minHandPresenceConfidence: Float = DEFAULT_PRESENCE_CONFIDENCE,
    val minTrackingConfidence: Float = DEFAULT_TRACKING_CONFIDENCE,
    internal val engine: HandLandmarkerEngine
) : IHandLandmarkDetector {

    /**
     * Primary Android constructor.
     * Initializes native HandLandmarker from assets, with graceful fallback to JVM stub on desktops.
     */
    constructor(
        context: Context,
        modelAssetPath: String = DEFAULT_MODEL_ASSET_PATH,
        numHands: Int = DEFAULT_NUM_HANDS,
        minHandDetectionConfidence: Float = DEFAULT_DETECTION_CONFIDENCE,
        minHandPresenceConfidence: Float = DEFAULT_PRESENCE_CONFIDENCE,
        minTrackingConfidence: Float = DEFAULT_TRACKING_CONFIDENCE
    ) : this(
        context = context,
        modelAssetPath = modelAssetPath,
        numHands = numHands,
        minHandDetectionConfidence = minHandDetectionConfidence,
        minHandPresenceConfidence = minHandPresenceConfidence,
        minTrackingConfidence = minTrackingConfidence,
        engine = createEngine(
            context = context,
            modelAssetPath = modelAssetPath,
            numHands = numHands,
            minHandDetectionConfidence = minHandDetectionConfidence,
            minHandPresenceConfidence = minHandPresenceConfidence,
            minTrackingConfidence = minTrackingConfidence
        )
    )

    private var _isClosed = false
    override val isClosed: Boolean get() = _isClosed
    override val isInitialized: Boolean get() = !_isClosed && engine.isAvailable

    override val lastConversionMs: Double
        get() = engine.lastConversionMs

    override val lastRawDetectMs: Double
        get() = engine.lastRawDetectMs

    override fun detect(bitmap: Bitmap, rotationDegrees: Int): List<DetectedHandCandidate> {
        check(!_isClosed) { "HandLandmarkDetector has been closed" }
        return try {
            val result = engine.detect(bitmap, rotationDegrees)
            convertResultToCandidates(result)
        } catch (e: Exception) {
            Log.e(TAG, "Detection error on Bitmap: ${e.message}", e)
            emptyList()
        }
    }

    override fun detect(imageProxy: ImageProxy): List<DetectedHandCandidate> {
        check(!_isClosed) { "HandLandmarkDetector has been closed" }
        return try {
            val result = engine.detect(imageProxy)
            convertResultToCandidates(result)
        } catch (e: Exception) {
            Log.e(TAG, "Detection error on ImageProxy: ${e.message}", e)
            emptyList()
        }
    }

    override fun detectDetailed(imageProxy: ImageProxy): HandDetectionResult {
        check(!_isClosed) { "HandLandmarkDetector has been closed" }
        val candidates = detect(imageProxy)
        return HandDetectionResult(
            candidates = candidates,
            conversionMs = engine.lastConversionMs,
            rawDetectMs = engine.lastRawDetectMs
        )
    }

    override fun close() {
        if (!_isClosed) {
            _isClosed = true
            try {
                engine.close()
            } catch (e: Exception) {
                Log.w(TAG, "Exception during detector close: ${e.message}", e)
            }
        }
    }

    companion object {
        private const val TAG = "HandLandmarkDetector"
        const val DEFAULT_MODEL_ASSET_PATH = "hand_landmarker.task"
        const val DEFAULT_NUM_HANDS = 2
        const val DEFAULT_DETECTION_CONFIDENCE = 0.5f
        const val DEFAULT_PRESENCE_CONFIDENCE = 0.5f
        const val DEFAULT_TRACKING_CONFIDENCE = 0.5f
        const val LANDMARK_COUNT = 21
        const val COORDINATE_DIM = 3
        const val HAND_FEATURE_DIM = 63 // 21 * 3, matches HandSlotResolver.HAND_FEATURE_DIM

        /**
         * Factory creating an engine with native MediaPipe on Android, or a fallback stub on JVM.
         */
        private fun createEngine(
            context: Context,
            modelAssetPath: String,
            numHands: Int,
            minHandDetectionConfidence: Float,
            minHandPresenceConfidence: Float,
            minTrackingConfidence: Float
        ): HandLandmarkerEngine {
            return try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath(modelAssetPath)
                    .build()

                val options = HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setNumHands(numHands)
                    .setMinHandDetectionConfidence(minHandDetectionConfidence)
                    .setMinHandPresenceConfidence(minHandPresenceConfidence)
                    .setMinTrackingConfidence(minTrackingConfidence)
                    .build()

                val landmarker = HandLandmarker.createFromOptions(context, options)
                NativeHandLandmarkerEngine(landmarker)
            } catch (e: LinkageError) {
                Log.w(TAG, "MediaPipe native JNI not loaded (host JVM environment): ${e.message}")
                HostJvmFallbackEngine()
            } catch (e: Exception) {
                Log.w(TAG, "MediaPipe initialization failed: ${e.message}", e)
                HostJvmFallbackEngine()
            }
        }

        /**
         * Converts MediaPipe's raw [HandLandmarkerResult] into EchoGesture's [DetectedHandCandidate] list.
         *
         * Enforces EchoGesture invariants:
         * - Clamps candidate count to at most 2 hands
         * - Flattens 21 3D landmarks into a FloatArray of size 63 [x0, y0, z0, ..., x20, y20, z20]
         * - Coordinates x and y are normalized to [0.0, 1.0], z represents depth relative to wrist
         * - Sets wristX = landmarks63[0], wristY = landmarks63[1]
         * - Maps handedness category to canonical "Left" or "Right" with confidence in [0.0, 1.0]
         * - Returns empty list for 0-hand or missing detection cases
         */
        fun convertResultToCandidates(result: HandLandmarkerResult?): List<DetectedHandCandidate> {
            if (result == null) return emptyList()

            val allLandmarks = try {
                result.landmarks()
            } catch (e: Exception) {
                null
            }
            if (allLandmarks.isNullOrEmpty()) return emptyList()

            val handednessList = try {
                result.handedness()
            } catch (e: Exception) {
                null
            } ?: emptyList()

            val maxHands = minOf(allLandmarks.size, DEFAULT_NUM_HANDS)
            val candidates = ArrayList<DetectedHandCandidate>(maxHands)

            for (h in 0 until maxHands) {
                val handLandmarks = allLandmarks[h]
                if (handLandmarks.isNullOrEmpty()) continue

                val landmarks63 = FloatArray(HAND_FEATURE_DIM)
                val count = minOf(handLandmarks.size, LANDMARK_COUNT)
                for (i in 0 until count) {
                    val lm = handLandmarks[i]
                    val base = i * COORDINATE_DIM
                    landmarks63[base] = lm.x()
                    landmarks63[base + 1] = lm.y()
                    landmarks63[base + 2] = lm.z()
                }

                val categories = handednessList.getOrNull(h)
                val primaryCategory = categories?.firstOrNull()
                val rawLabel = primaryCategory?.categoryName()?.trim() ?: ""
                val label = when (rawLabel.lowercase()) {
                    "left" -> "Left"
                    "right" -> "Right"
                    else -> if (rawLabel.isNotEmpty()) rawLabel else "Unknown"
                }
                val score = (primaryCategory?.score() ?: 0.0f).coerceIn(0.0f, 1.0f)

                candidates.add(
                    DetectedHandCandidate(
                        label = label,
                        score = score,
                        landmarks63 = landmarks63,
                        wristX = landmarks63[0],
                        wristY = landmarks63[1]
                    )
                )
            }

            return candidates
        }

        /**
         * Factory function creating a detector with a mock/fallback engine for unit testing.
         */
        fun createForTesting(
            testResult: HandLandmarkerResult? = null
        ): HandLandmarkDetector {
            val fallback = HostJvmFallbackEngine()
            if (testResult != null) {
                fallback.setTestResult(testResult)
            }
            return HandLandmarkDetector(
                context = null,
                engine = fallback
            )
        }
    }
}
