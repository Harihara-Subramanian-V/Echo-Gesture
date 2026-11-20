package com.echogesture.app.camera

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Functional listener receiving unmirrored canonical camera frames.
 */
fun interface CameraFrameListener {
    fun onFrameCaptured(imageProxy: ImageProxy)
}

/**
 * Functional listener receiving camera lifecycle and diagnostic status changes.
 */
fun interface CameraStatusListener {
    fun onStatusChanged(status: String)
}

/**
 * Abstraction over CameraProvider lifecycle binding for unit testability and hardware decoupling.
 */
interface CameraProviderAdapter {
    fun hasCamera(cameraSelector: CameraSelector): Boolean
    fun unbindAll()
    fun bindToLifecycle(
        lifecycleOwner: LifecycleOwner,
        cameraSelector: CameraSelector,
        preview: Preview?,
        imageAnalysis: ImageAnalysis?
    )
}

/**
 * Default production implementation backed by CameraX [ProcessCameraProvider].
 */
class DefaultCameraProviderAdapter(
    private val provider: ProcessCameraProvider
) : CameraProviderAdapter {
    override fun hasCamera(cameraSelector: CameraSelector): Boolean {
        return try {
            provider.hasCamera(cameraSelector)
        } catch (_: Exception) {
            false
        }
    }

    override fun unbindAll() {
        provider.unbindAll()
    }

    override fun bindToLifecycle(
        lifecycleOwner: LifecycleOwner,
        cameraSelector: CameraSelector,
        preview: Preview?,
        imageAnalysis: ImageAnalysis?
    ) {
        val useCases = mutableListOf<UseCase>()
        if (preview != null) useCases.add(preview)
        if (imageAnalysis != null) useCases.add(imageAnalysis)
        if (useCases.isNotEmpty()) {
            provider.bindToLifecycle(lifecycleOwner, cameraSelector, *useCases.toTypedArray())
        }
    }
}

/**
 * CameraX Capture Controller for EchoGesture real-time recognition.
 *
 * Key Invariants:
 * - Lens Facing: Default to Rear Camera ([CameraSelector.DEFAULT_BACK_CAMERA]), switchable via [toggleCamera].
 * - Target Resolution: 640x480 configured via [ResolutionSelector] with
 *   [ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER].
 * - Backpressure: [ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST] drops stale frames while analysis is busy.
 * - Threading: Offloads image analysis onto a dedicated single-thread [ExecutorService].
 * - Frame Canonicality: Delivers raw, unmirrored canonical [ImageProxy] buffers to [CameraFrameListener]
 *   regardless of camera lens facing. UI preview mirroring is delegated solely to [PreviewView].
 * - Buffer Safety: Strictly closes every [ImageProxy] in a `finally` block to prevent CameraX buffer starvation.
 * - Lifecycle Controls: Provides explicit [start], [pause], [resume], [stop], and [shutdown] controls.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private var previewView: PreviewView? = null,
    var frameListener: CameraFrameListener? = null,
    var statusListener: CameraStatusListener? = null,
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor(),
    private var cameraProviderAdapter: CameraProviderAdapter? = null
) {
    var currentLensFacing: Int = CameraSelector.LENS_FACING_BACK
        private set

    val isFrontCamera: Boolean
        get() = currentLensFacing == CameraSelector.LENS_FACING_FRONT

    val cameraSelector: CameraSelector
        get() = if (isFrontCamera) {
            CameraSelector.DEFAULT_FRONT_CAMERA
        } else {
            CameraSelector.DEFAULT_BACK_CAMERA
        }

    val targetResolution: Size = TARGET_RESOLUTION

    var isAnalyzing: Boolean = false
        private set

    var isBound: Boolean = false
        private set

    var isShutdown: Boolean = false
        private set

    fun attachPreviewView(previewView: PreviewView) {
        this.previewView = previewView
        if (isBound && !isShutdown) {
            bindCameraUseCases()
        }
    }

    fun start() {
        if (isShutdown) {
            statusListener?.onStatusChanged("Camera Error: Controller is shut down")
            return
        }

        val adapter = cameraProviderAdapter
        if (adapter != null) {
            bindCameraUseCases()
            return
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProviderAdapter = DefaultCameraProviderAdapter(provider)
                bindCameraUseCases()
            } catch (e: Exception) {
                statusListener?.onStatusChanged("Camera Error: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun toggleCamera() {
        if (isShutdown) return
        currentLensFacing = if (currentLensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        val facingName = if (isFrontCamera) "Front" else "Rear"
        statusListener?.onStatusChanged("Camera: Switched to $facingName")
        if (isBound) {
            bindCameraUseCases()
        }
    }

    fun setLensFacing(lensFacing: Int) {
        if (isShutdown || currentLensFacing == lensFacing) return
        currentLensFacing = lensFacing
        val facingName = if (isFrontCamera) "Front" else "Rear"
        statusListener?.onStatusChanged("Camera: Switched to $facingName")
        if (isBound) {
            bindCameraUseCases()
        }
    }

    fun pause() {
        if (isShutdown) return
        isAnalyzing = false
        statusListener?.onStatusChanged("Camera: Paused")
    }

    fun resume() {
        if (isShutdown) return
        isAnalyzing = true
        statusListener?.onStatusChanged("Camera: Resumed")
    }

    fun stop() {
        if (isShutdown) return
        isAnalyzing = false
        isBound = false
        cameraProviderAdapter?.unbindAll()
        statusListener?.onStatusChanged("Camera: Stopped")
    }

    fun shutdown() {
        if (isShutdown) return
        stop()
        isShutdown = true
        analysisExecutor.shutdown()
        statusListener?.onStatusChanged("Camera: Shutdown")
        frameListener = null
        statusListener = null
    }

    fun buildResolutionSelector(): ResolutionSelector {
        return ResolutionSelector.Builder()
            .setResolutionStrategy(
                ResolutionStrategy(
                    targetResolution,
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()
    }

    fun buildPreview(resolutionSelector: ResolutionSelector = buildResolutionSelector()): Preview? {
        val pv = previewView ?: return null
        return Preview.Builder()
            .setResolutionSelector(resolutionSelector)
            .build()
            .also { preview ->
                preview.surfaceProvider = pv.surfaceProvider
            }
    }

    fun buildImageAnalysis(resolutionSelector: ResolutionSelector = buildResolutionSelector()): ImageAnalysis? {
        return try {
            ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
        } catch (_: Throwable) {
            // Null fallback during uninitialized JVM environment
            null
        }
    }

    internal fun bindCameraUseCases() {
        val adapter = cameraProviderAdapter ?: return
        try {
            adapter.unbindAll()

            val resolutionSelector = buildResolutionSelector()
            val preview = buildPreview(resolutionSelector)
            val imageAnalysis = buildImageAnalysis(resolutionSelector)

            imageAnalysis?.setAnalyzer(analysisExecutor, createAnalyzer())

            val selector = cameraSelector
            if (adapter.hasCamera(selector)) {
                adapter.bindToLifecycle(lifecycleOwner, selector, preview, imageAnalysis)
                isBound = true
                isAnalyzing = true
                val facingName = if (isFrontCamera) "Front" else "Rear"
                statusListener?.onStatusChanged("Camera: Active $facingName (${TARGET_WIDTH}x${TARGET_HEIGHT})")
            } else {
                isBound = false
                isAnalyzing = false
                statusListener?.onStatusChanged("Camera Error: Camera selector unavailable")
            }
        } catch (e: Exception) {
            isBound = false
            isAnalyzing = false
            statusListener?.onStatusChanged("Bind Failed: ${e.message}")
        }
    }

    internal fun createAnalyzer(): ImageAnalysis.Analyzer {
        return ImageAnalysis.Analyzer { imageProxy ->
            processFrame(imageProxy)
        }
    }

    internal fun processFrame(imageProxy: ImageProxy) {
        try {
            if (isAnalyzing && !isShutdown) {
                frameListener?.onFrameCaptured(imageProxy)
            }
        } catch (t: Throwable) {
            statusListener?.onStatusChanged("Frame Analysis Error: ${t.message}")
        } finally {
            imageProxy.close()
        }
    }

    companion object {
        const val TARGET_WIDTH = 640
        const val TARGET_HEIGHT = 480
        val TARGET_RESOLUTION = Size(TARGET_WIDTH, TARGET_HEIGHT)
    }
}
