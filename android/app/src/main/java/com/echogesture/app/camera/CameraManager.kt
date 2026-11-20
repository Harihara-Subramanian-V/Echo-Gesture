package com.echogesture.app.camera

import android.content.Context
import androidx.camera.core.ImageProxy
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner

/**
 * Backward-compatible facade delegating to [CameraController].
 */
class CameraManager(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    private val onFrameReceived: (ImageProxy) -> Unit,
    private val onStatusUpdate: (String) -> Unit
) {
    val controller = CameraController(
        context = context,
        lifecycleOwner = lifecycleOwner,
        previewView = previewView,
        frameListener = { onFrameReceived(it) },
        statusListener = { onStatusUpdate(it) }
    )

    val isAnalyzing: Boolean get() = controller.isAnalyzing
    val isFrontCamera: Boolean get() = controller.isFrontCamera
    val currentLensFacing: Int get() = controller.currentLensFacing

    fun start() = controller.start()
    fun pause() = controller.pause()
    fun resume() = controller.resume()
    fun stop() = controller.stop()
    fun release() = controller.shutdown()
    fun toggleCamera() = controller.toggleCamera()
    fun setLensFacing(facing: Int) = controller.setLensFacing(facing)
}
