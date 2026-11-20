package com.echogesture.app.vision

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import com.echogesture.app.feature.DetectedHandCandidate

/**
 * Result of hand landmark detection with explicit stage-wise timing.
 */
data class HandDetectionResult(
    val candidates: List<DetectedHandCandidate>,
    val conversionMs: Double = 0.0,
    val rawDetectMs: Double = 0.0
)

/**
 * Contract for hand landmark detection, abstracting the vision pipeline from the underlying
 * inference engine (Google MediaPipe Tasks Vision or test stubs).
 *
 * Adheres strictly to EchoGesture Phase 7C specifications:
 * - 21 normalized landmarks per hand flattened to a 63-D array: [x0, y0, z0, ..., x20, y20, z20]
 * - Handedness classification with score in [0.0, 1.0] and canonical label ("Left" / "Right")
 * - Up to 2 concurrent hands detected per frame
 * - Synchronous per-frame execution matching CameraX ImageAnalysis pacing
 */
interface IHandLandmarkDetector : AutoCloseable {

    /**
     * Whether the detector engine is loaded and ready to process images.
     */
    val isInitialized: Boolean

    /**
     * Whether the detector has been closed and resources released.
     */
    val isClosed: Boolean

    /**
     * Latency of frame conversion (e.g. ImageProxy to Bitmap / MPImage) in milliseconds.
     */
    val lastConversionMs: Double get() = 0.0

    /**
     * Latency of raw MediaPipe hand landmark inference in milliseconds.
     */
    val lastRawDetectMs: Double get() = 0.0

    /**
     * Synchronously detects hand landmarks from an Android [Bitmap].
     *
     * @param bitmap Input bitmap image in ARGB_8888 or RGB_565 format.
     * @param rotationDegrees Counter-clockwise rotation angle in degrees (0, 90, 180, 270).
     * @return List of detected hand candidates, up to 2 hands. Empty if no hands detected.
     */
    fun detect(bitmap: Bitmap, rotationDegrees: Int = 0): List<DetectedHandCandidate>

    /**
     * Synchronously detects hand landmarks from a CameraX [ImageProxy].
     * Rotation degrees are automatically extracted from [ImageProxy.getImageInfo].
     *
     * Note: The caller or enclosing analyzer retains ownership of the ImageProxy lifecycle;
     * detect() does not close the ImageProxy.
     *
     * @param imageProxy CameraX frame buffer.
     * @return List of detected hand candidates, up to 2 hands. Empty if no hands detected.
     */
    fun detect(imageProxy: ImageProxy): List<DetectedHandCandidate>

    /**
     * Synchronously detects hand landmarks with explicit stage-wise timing.
     *
     * @param imageProxy CameraX frame buffer.
     * @return [HandDetectionResult] containing candidates and conversion/rawDetect timings.
     */
    fun detectDetailed(imageProxy: ImageProxy): HandDetectionResult {
        val candidates = detect(imageProxy)
        return HandDetectionResult(
            candidates = candidates,
            conversionMs = lastConversionMs,
            rawDetectMs = lastRawDetectMs
        )
    }

    /**
     * Releases underlying MediaPipe TaskRunner and native pipeline resources.
     */
    override fun close()
}
