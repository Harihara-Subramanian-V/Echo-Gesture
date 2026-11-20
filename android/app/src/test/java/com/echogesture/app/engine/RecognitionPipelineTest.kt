package com.echogesture.app.engine

import android.content.Context
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.echogesture.app.feature.DetectedHandCandidate
import com.echogesture.app.feature.FeatureExtractor134D
import com.echogesture.app.feature.HandSlotResolver
import com.echogesture.app.inference.LiteRtModelRunner
import com.echogesture.app.preprocessing.SequenceNormalizer
import com.echogesture.app.preprocessing.SequencePreprocessor263D
import com.echogesture.app.stabilization.RecognitionState
import com.echogesture.app.vision.IHandLandmarkDetector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.lang.reflect.Proxy

class RecognitionPipelineTest {

    private lateinit var pipeline: RecognitionPipeline
    private var lastDispatchedState: PipelineUiState? = null

    // Mock detector that returns configurable hands
    private var mockDetectedHands: List<DetectedHandCandidate> = emptyList()

    private val stubDetector = object : IHandLandmarkDetector {
        override val isClosed: Boolean = false
        override val isInitialized: Boolean = true

        override fun detect(bitmap: android.graphics.Bitmap, rotationDegrees: Int): List<DetectedHandCandidate> {
            return mockDetectedHands
        }

        override fun detect(imageProxy: ImageProxy): List<DetectedHandCandidate> {
            return mockDetectedHands
        }

        override fun close() {}
    }

    private fun findAssetFile(fileName: String): File {
        val candidates = listOf(
            File("src/main/assets", fileName),
            File("app/src/main/assets", fileName),
            File("../app/src/main/assets", fileName),
            File("../../app/src/main/assets", fileName),
            File("android/app/src/main/assets", fileName)
        )
        return candidates.firstOrNull { it.exists() } ?: File("android/app/src/main/assets", fileName)
    }

    @Before
    fun setUp() {
        val mean = FloatArray(134) { 0.0f }
        val std = FloatArray(134) { 1.0f }
        val normalizer = SequenceNormalizer(mean, std)
        val preprocessor = SequencePreprocessor263D(normalizer)

        val runner = LiteRtModelRunner()
        val modelFile = findAssetFile("echogesture_exp_d_int8.tflite")
        val vocabFile = findAssetFile("label_encoder.json")
        if (modelFile.exists() && vocabFile.exists()) {
            runner.initializeFromFile(modelFile, vocabFile, numThreads = 2)
        } else {
            runner.initialize()
        }

        pipeline = RecognitionPipeline(
            detector = stubDetector,
            modelRunner = runner,
            preprocessor = preprocessor
        )

        pipeline.stateListener = PipelineStateListener { state ->
            lastDispatchedState = state
        }
    }

    private fun createDummyImageProxy(w: Int = 640, h: Int = 480): ImageProxy {
        return Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getWidth" -> w
                "getHeight" -> h
                "getFormat" -> 35 // YUV_420_888
                "getImageInfo" -> {
                    Proxy.newProxyInstance(
                        ImageInfo::class.java.classLoader,
                        arrayOf(ImageInfo::class.java)
                    ) { _, infoMethod, _ ->
                        when (infoMethod.name) {
                            "getRotationDegrees" -> 0
                            else -> null
                        }
                    }
                }
                "close" -> null
                else -> null
            }
        } as ImageProxy
    }

    private fun createDummyHand(label: String, wristX: Float, wristY: Float): DetectedHandCandidate {
        val landmarks = FloatArray(63)
        landmarks[0] = wristX
        landmarks[1] = wristY
        landmarks[2] = 0.0f
        return DetectedHandCandidate(
            label = label,
            score = 0.95f,
            landmarks63 = landmarks,
            wristX = wristX,
            wristY = wristY
        )
    }

    @Test
    fun testInitialStateDispatchesWaitingForHands() {
        mockDetectedHands = emptyList()
        val frame = createDummyImageProxy()

        pipeline.processFrame(frame, isFrontCamera = false)

        assertNotNull(lastDispatchedState)
        assertEquals(RecognitionState.WAITING_FOR_HANDS, lastDispatchedState?.state)
        assertEquals(0, lastDispatchedState?.detectedHands?.size)
        assertTrue(lastDispatchedState?.latencies?.totalMs ?: -1.0 >= 0.0)
    }

    @Test
    fun testCollectingFramesWhenHandsDetected() {
        mockDetectedHands = listOf(createDummyHand("Right", 0.4f, 0.5f))
        val frame = createDummyImageProxy()

        pipeline.processFrame(frame, isFrontCamera = false)

        assertNotNull(lastDispatchedState)
        assertEquals(RecognitionState.COLLECTING_FRAMES, lastDispatchedState?.state)
        assertEquals(1, lastDispatchedState?.detectedHands?.size)
        assertEquals(1, lastDispatchedState?.bufferFill)
    }

    @Test
    fun testPipelineTransitionsToInferenceAfterMinFrames() {
        mockDetectedHands = listOf(
            createDummyHand("Right", 0.4f, 0.5f),
            createDummyHand("Left", 0.6f, 0.5f)
        )
        val frame = createDummyImageProxy()

        // Feed 16 frames to meet minFramesForInference
        for (i in 0 until 16) {
            pipeline.processFrame(frame, isFrontCamera = false)
        }

        assertNotNull(lastDispatchedState)
        // With 16 frames accumulated, buffer initiates inference and stabilization
        assertTrue(
            lastDispatchedState?.state == RecognitionState.RECOGNIZING ||
            lastDispatchedState?.state == RecognitionState.ACTIVE_PREDICTION
        )
        assertEquals(16, lastDispatchedState?.bufferFill)
        assertTrue(lastDispatchedState?.latencies?.inferenceMs ?: -1.0 >= 0.0)
    }

    @Test
    fun testResetClearsStateAndBuffers() {
        mockDetectedHands = listOf(createDummyHand("Right", 0.4f, 0.5f))
        pipeline.processFrame(createDummyImageProxy(), isFrontCamera = false)

        pipeline.reset()

        assertEquals(0, pipeline.temporalBuffer.size)
        assertEquals(RecognitionState.WAITING_FOR_HANDS, pipeline.stabilizer.currentState().state)
    }

    @Test
    fun testCloseReleasesComponentsSafely() {
        pipeline.close()
        // Subsequent calls should safely no-op
        pipeline.processFrame(createDummyImageProxy(), isFrontCamera = false)
        // No exception thrown
    }
}
