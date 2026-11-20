package com.echogesture.app.camera

import android.content.Context
import android.graphics.Rect
import android.media.Image
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.impl.TagBundle
import androidx.camera.core.impl.utils.ExifData
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.Executors

class CameraControllerTest {

    private lateinit var fakeLifecycleOwner: FakeLifecycleOwner
    private lateinit var fakeProviderAdapter: FakeCameraProviderAdapter
    private lateinit var dummyContext: Context
    private lateinit var controller: CameraController
    private lateinit var statusMessages: MutableList<String>

    @Before
    fun setUp() {
        fakeLifecycleOwner = FakeLifecycleOwner()
        fakeProviderAdapter = FakeCameraProviderAdapter()
        dummyContext = android.content.ContextWrapper(null)
        statusMessages = mutableListOf()

        controller = CameraController(
            context = dummyContext,
            lifecycleOwner = fakeLifecycleOwner,
            previewView = null,
            statusListener = { statusMessages.add(it) },
            analysisExecutor = Executors.newSingleThreadExecutor(),
            cameraProviderAdapter = fakeProviderAdapter
        )
    }

    // -------------------------------------------------------------------------
    // 1. Camera Selector & Lens Facing Toggling
    // -------------------------------------------------------------------------

    @Test
    fun testDefaultCameraIsRear() {
        assertEquals(CameraSelector.LENS_FACING_BACK, controller.currentLensFacing)
        assertFalse(controller.isFrontCamera)
        assertEquals(CameraSelector.DEFAULT_BACK_CAMERA, controller.cameraSelector)
    }

    @Test
    fun testToggleCameraSwitchesBetweenRearAndFront() {
        // Rear -> Front
        controller.toggleCamera()
        assertEquals(CameraSelector.LENS_FACING_FRONT, controller.currentLensFacing)
        assertTrue(controller.isFrontCamera)
        assertEquals(CameraSelector.DEFAULT_FRONT_CAMERA, controller.cameraSelector)
        assertTrue(statusMessages.last().contains("Switched to Front"))

        // Front -> Rear
        controller.toggleCamera()
        assertEquals(CameraSelector.LENS_FACING_BACK, controller.currentLensFacing)
        assertFalse(controller.isFrontCamera)
        assertEquals(CameraSelector.DEFAULT_BACK_CAMERA, controller.cameraSelector)
        assertTrue(statusMessages.last().contains("Switched to Rear"))
    }

    @Test
    fun testSetLensFacingExplicitly() {
        controller.setLensFacing(CameraSelector.LENS_FACING_FRONT)
        assertTrue(controller.isFrontCamera)
        assertEquals(CameraSelector.DEFAULT_FRONT_CAMERA, controller.cameraSelector)

        // Idempotent call with same facing should not duplicate
        val countBefore = statusMessages.size
        controller.setLensFacing(CameraSelector.LENS_FACING_FRONT)
        assertEquals(countBefore, statusMessages.size)

        controller.setLensFacing(CameraSelector.LENS_FACING_BACK)
        assertFalse(controller.isFrontCamera)
        assertEquals(CameraSelector.DEFAULT_BACK_CAMERA, controller.cameraSelector)
    }

    // -------------------------------------------------------------------------
    // 2. Resolution & Backpressure Strategy Configuration
    // -------------------------------------------------------------------------

    @Test
    fun testTargetResolutionAndStrategy() {
        assertEquals(640, CameraController.TARGET_WIDTH)
        assertEquals(480, CameraController.TARGET_HEIGHT)
        assertSame(CameraController.TARGET_RESOLUTION, controller.targetResolution)

        val resolutionSelector = controller.buildResolutionSelector()
        assertNotNull(resolutionSelector)
        val strategy = resolutionSelector.resolutionStrategy
        assertNotNull(strategy)
        assertSame(controller.targetResolution, strategy!!.boundSize)
        assertEquals(
            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
            strategy.fallbackRule
        )
    }

    @Test
    fun testImageAnalysisContractSpecification() {
        // Verify contract constants required for EchoGesture recognition pipeline
        assertEquals(
            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST,
            0 // KEEP_ONLY_LATEST constant value in CameraX
        )
        assertEquals(
            ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888,
            2 // OUTPUT_IMAGE_FORMAT_RGBA_8888 constant value
        )
        assertEquals(640, CameraController.TARGET_WIDTH)
        assertEquals(480, CameraController.TARGET_HEIGHT)
    }

    // -------------------------------------------------------------------------
    // 3. Pause, Resume, Stop, and Shutdown Lifecycle Invariants
    // -------------------------------------------------------------------------

    @Test
    fun testPauseAndResumeInvariants() {
        assertFalse(controller.isAnalyzing)
        assertFalse(controller.isShutdown)

        controller.resume()
        assertTrue(controller.isAnalyzing)
        assertTrue(statusMessages.last().contains("Resumed"))

        controller.pause()
        assertFalse(controller.isAnalyzing)
        assertTrue(statusMessages.last().contains("Paused"))

        controller.resume()
        assertTrue(controller.isAnalyzing)
    }

    @Test
    fun testStopUnbindsAndResetsState() {
        controller.start()
        assertTrue(controller.isBound)
        assertTrue(controller.isAnalyzing)

        controller.stop()
        assertFalse(controller.isBound)
        assertFalse(controller.isAnalyzing)
        assertTrue(statusMessages.last().contains("Stopped"))
        assertTrue(fakeProviderAdapter.unbindAllCount >= 1)
    }

    @Test
    fun testShutdownPermanentTermination() {
        controller.start()
        assertTrue(controller.isBound)

        controller.shutdown()
        assertTrue(controller.isShutdown)
        assertFalse(controller.isBound)
        assertFalse(controller.isAnalyzing)
        assertTrue(statusMessages.contains("Camera: Shutdown"))

        // Subsequent controls must be safely ignored when shutdown
        controller.resume()
        assertFalse(controller.isAnalyzing)

        controller.toggleCamera()
        assertEquals(CameraSelector.LENS_FACING_BACK, controller.currentLensFacing)

        controller.start()
        assertFalse(controller.isBound)
    }

    // -------------------------------------------------------------------------
    // 4. Frame Listener Dispatch & Strict ImageProxy Buffer Cleanup
    // -------------------------------------------------------------------------

    @Test
    fun testFrameDispatchedAndClosedWhenAnalyzing() {
        var capturedProxy: ImageProxy? = null
        controller.frameListener = CameraFrameListener { proxy ->
            capturedProxy = proxy
        }
        controller.resume()
        assertTrue(controller.isAnalyzing)

        val fakeProxy = FakeImageProxy(640, 480)
        controller.processFrame(fakeProxy)

        assertSame(fakeProxy, capturedProxy)
        assertTrue("ImageProxy must be closed after dispatch", fakeProxy.isClosed)
        assertEquals(1, fakeProxy.closeCallCount)
    }

    @Test
    fun testFrameDroppedAndClosedWhenPaused() {
        var dispatched = false
        controller.frameListener = CameraFrameListener {
            dispatched = true
        }
        controller.pause()
        assertFalse(controller.isAnalyzing)

        val fakeProxy = FakeImageProxy(640, 480)
        controller.processFrame(fakeProxy)

        assertFalse("Listener must not receive frame when paused", dispatched)
        assertTrue("ImageProxy must still be closed when paused", fakeProxy.isClosed)
        assertEquals(1, fakeProxy.closeCallCount)
    }

    @Test
    fun testFrameDroppedAndClosedWhenShutdown() {
        var dispatched = false
        controller.frameListener = CameraFrameListener {
            dispatched = true
        }
        controller.shutdown()

        val fakeProxy = FakeImageProxy(640, 480)
        controller.processFrame(fakeProxy)

        assertFalse("Listener must not receive frame after shutdown", dispatched)
        assertTrue("ImageProxy must still be closed after shutdown", fakeProxy.isClosed)
        assertEquals(1, fakeProxy.closeCallCount)
    }

    @Test
    fun testFrameListenerExceptionHandledAndProxyClosedSafely() {
        controller.frameListener = CameraFrameListener {
            throw RuntimeException("Simulated ML inference pipeline fault")
        }
        controller.resume()

        val fakeProxy = FakeImageProxy(640, 480)
        controller.processFrame(fakeProxy)

        assertTrue("ImageProxy must be closed even if listener throws", fakeProxy.isClosed)
        assertEquals(1, fakeProxy.closeCallCount)
        assertTrue(
            "Status listener must report the error message",
            statusMessages.last().contains("Simulated ML inference pipeline fault")
        )
    }

    // -------------------------------------------------------------------------
    // 5. Provider Binding & Camera Re-binding on Toggle
    // -------------------------------------------------------------------------

    @Test
    fun testBindingWithCameraProviderAdapter() {
        controller.start()
        assertTrue(controller.isBound)
        assertTrue(controller.isAnalyzing)
        assertEquals(1, fakeProviderAdapter.bindCount)
        assertEquals(CameraSelector.DEFAULT_BACK_CAMERA, fakeProviderAdapter.lastCameraSelector)

        // Toggling while bound re-binds with front camera
        controller.toggleCamera()
        assertEquals(2, fakeProviderAdapter.bindCount)
        assertEquals(CameraSelector.DEFAULT_FRONT_CAMERA, fakeProviderAdapter.lastCameraSelector)
    }

    @Test
    fun testUnavailableCameraSelectorFailsGracefully() {
        fakeProviderAdapter.hasCameraReturn = false
        controller.start()

        assertFalse(controller.isBound)
        assertFalse(controller.isAnalyzing)
        assertTrue(statusMessages.last().contains("unavailable"))
    }

    // -------------------------------------------------------------------------
    // Test Fakes & Helpers
    // -------------------------------------------------------------------------

    private class FakeLifecycle : Lifecycle() {
        override val currentState: State = State.RESUMED
        override fun addObserver(observer: androidx.lifecycle.LifecycleObserver) {}
        override fun removeObserver(observer: androidx.lifecycle.LifecycleObserver) {}
    }

    private class FakeLifecycleOwner(
        override val lifecycle: Lifecycle = FakeLifecycle()
    ) : LifecycleOwner

    private class FakeCameraProviderAdapter(
        var hasCameraReturn: Boolean = true
    ) : CameraProviderAdapter {
        var unbindAllCount: Int = 0
        var bindCount: Int = 0
        var lastCameraSelector: CameraSelector? = null
        var lastLifecycleOwner: LifecycleOwner? = null

        override fun hasCamera(cameraSelector: CameraSelector): Boolean = hasCameraReturn

        override fun unbindAll() {
            unbindAllCount++
        }

        override fun bindToLifecycle(
            lifecycleOwner: LifecycleOwner,
            cameraSelector: CameraSelector,
            preview: Preview?,
            imageAnalysis: ImageAnalysis?
        ) {
            bindCount++
            lastLifecycleOwner = lifecycleOwner
            lastCameraSelector = cameraSelector
        }
    }

    private class FakeImageProxy(
        private val widthVal: Int = 640,
        private val heightVal: Int = 480,
        private val formatVal: Int = 35 // ImageFormat.YUV_420_888
    ) : ImageProxy {
        var isClosed: Boolean = false
            private set
        var closeCallCount: Int = 0
            private set

        override fun close() {
            isClosed = true
            closeCallCount++
        }

        override fun getCropRect(): Rect = Rect(0, 0, widthVal, heightVal)
        override fun setCropRect(rect: Rect?) {}
        override fun getFormat(): Int = formatVal
        override fun getHeight(): Int = heightVal
        override fun getWidth(): Int = widthVal
        override fun getPlanes(): Array<ImageProxy.PlaneProxy> = emptyArray()
        override fun getImageInfo(): ImageInfo = object : ImageInfo {
            override fun getTagBundle(): TagBundle = TagBundle.emptyBundle()
            override fun getTimestamp(): Long = System.nanoTime()
            override fun getRotationDegrees(): Int = 0
            override fun populateExifData(builder: ExifData.Builder) {}
        }
        override fun getImage(): Image? = null
    }
}
