package com.echogesture.app.vision

import android.graphics.Bitmap
import androidx.camera.core.ImageProxy
import com.echogesture.app.feature.DetectedHandCandidate

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
     * Releases underlying MediaPipe TaskRunner and native pipeline resources.
     */
    override fun close()
}
