package com.echogesture.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Choreographer
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.echogesture.app.R
import com.echogesture.app.camera.CameraController
import com.echogesture.app.databinding.ActivityMainBinding
import com.echogesture.app.engine.PipelineUiState
import com.echogesture.app.engine.PipelineStateListener
import com.echogesture.app.engine.RecognitionPipeline
import com.echogesture.app.inference.LiteRtModelRunner
import com.echogesture.app.preprocessing.SequenceNormalizer
import com.echogesture.app.preprocessing.SequencePreprocessor263D
import com.echogesture.app.speech.PredictionSpeechCoordinator
import com.echogesture.app.speech.SpeechEventListener
import com.echogesture.app.speech.TextToSpeechManager
import com.echogesture.app.stabilization.RecognitionState
import com.echogesture.app.vision.HandLandmarkDetector
import java.util.Locale

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private var cameraController: CameraController? = null
    private var pipeline: RecognitionPipeline? = null
    private var ttsManager: TextToSpeechManager? = null
    private var speechCoordinator: PredictionSpeechCoordinator? = null
    private var isCapturing = true

    // Real-time Preview / Display Rendering FPS measurement
    private var previewFrameCount = 0
    private var lastPreviewFpsTimestamp = 0L
    private var previewFps = 0.0

    private val frameCallback = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            previewFrameCount++
            val nowMs = SystemClock.elapsedRealtime()
            if (lastPreviewFpsTimestamp == 0L) {
                lastPreviewFpsTimestamp = nowMs
            } else {
                val span = nowMs - lastPreviewFpsTimestamp
                if (span >= 1000L) {
                    previewFps = previewFrameCount * 1000.0 / span
                    Log.d("PREVIEW_FPS", "PREVIEW_FPS: previewFps=${String.format(Locale.US, "%.1f", previewFps)}")
                    previewFrameCount = 0
                    lastPreviewFpsTimestamp = nowMs
                }
            }
            if (!isFinishing && !isDestroyed) {
                Choreographer.getInstance().postFrameCallback(this)
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            binding.permissionContainer.visibility = View.GONE
            startCamera()
        } else {
            binding.permissionContainer.visibility = View.VISIBLE
            binding.tvCameraStatus.text = getString(R.string.label_camera_permission_required)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initializePipeline()
        initializeSpeech()
        setupListeners()
        checkCameraPermission()
        handleIntent(intent)
    }

    private fun initializeSpeech() {
        val tts = TextToSpeechManager(this)
        val coordinator = PredictionSpeechCoordinator(tts)
        this.ttsManager = tts
        this.speechCoordinator = coordinator

        tts.setSpeechEventListener(object : SpeechEventListener {
            override fun onSpeechStart(utteranceId: String) {
                runOnUiThread { this@MainActivity.updateSpeechStatusUi() }
            }

            override fun onSpeechDone(utteranceId: String) {
                runOnUiThread { this@MainActivity.updateSpeechStatusUi() }
            }

            override fun onSpeechError(utteranceId: String, errorCode: Int) {
                runOnUiThread { this@MainActivity.updateSpeechStatusUi() }
            }
        })

        tts.initialize { success ->
            runOnUiThread {
                this@MainActivity.updateSpeechStatusUi()
            }
        }
    }

    override fun onNewIntent(intent: android.content.Intent?) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: android.content.Intent?) {
        if (intent?.hasExtra("benchmark") == true) {
            val frames = intent.getIntExtra("frames", 500)
            runDetailedBenchmark(frames)
        }
    }

    private fun runDetailedBenchmark(targetFrames: Int = 500) {
        val p = pipeline ?: return
        Thread {
            Log.i(TAG, "STARTING_BENCHMARK: targetFrames=$targetFrames")

            // 1. Vision Benchmark on 640x480 Bitmap (50 iterations)
            try {
                val bmp = android.graphics.Bitmap.createBitmap(640, 480, android.graphics.Bitmap.Config.ARGB_8888)
                for (j in 1..50) {
                    val t0 = System.nanoTime()
                    p.detector.detect(bmp)
                    val t1 = System.nanoTime()
                    val totalMs = (t1 - t0) / 1_000_000.0
                    Log.i(
                        TAG,
                        "BENCH_VISION: iter=$j convMs=${String.format(Locale.US, "%.2f", p.detector.lastConversionMs)} " +
                        "rawDetectMs=${String.format(Locale.US, "%.2f", p.detector.lastRawDetectMs)} " +
                        "totalMs=${String.format(Locale.US, "%.2f", totalMs)}"
                    )
                }
                bmp.recycle()
            } catch (e: Exception) {
                Log.w(TAG, "Vision benchmark warning: ${e.message}")
            }

            // 2. Full Recognition & Inference Loop Benchmark (targetFrames iterations)
            val dummyCandidate = com.echogesture.app.feature.DetectedHandCandidate(
                label = "Right",
                score = 0.95f,
                landmarks63 = FloatArray(63) { (it % 21) * 0.04f + 0.1f },
                wristX = 0.5f,
                wristY = 0.5f
            )
            val candidates = listOf(dummyCandidate)

            for (i in 1..targetFrames) {
                val tStart = System.nanoTime()

                // Stage 2: Slot Resolution + 134-D Feature Extraction
                val tFeat0 = System.nanoTime()
                val slots = p.slotResolver.resolve(candidates)
                val frame134 = p.featureExtractor.extract(slots, 640, 480)
                val tFeat1 = System.nanoTime()
                val featMs = (tFeat1 - tFeat0) / 1_000_000.0

                // Stage 3: Temporal Buffer FIFO
                p.temporalBuffer.addFrame(frame134, true)
                val count = p.temporalBuffer.size

                var resampleMs = 0.0
                var prepMs = 0.0
                var quantMs = 0.0
                var modelMs = 0.0
                var decMs = 0.0
                var infMs = 0.0
                var stabMs = 0.0

                if (count >= p.stabilizer.minFramesForInference) {
                    val tRes0 = System.nanoTime()
                    val snapshot = p.temporalBuffer.getResampledSnapshot(64)
                    val tRes1 = System.nanoTime()
                    resampleMs = (tRes1 - tRes0) / 1_000_000.0

                    val tInf0 = System.nanoTime()
                    val detInf = p.inferenceEngine.predictDetailed(snapshot)
                    val tInf1 = System.nanoTime()
                    infMs = (tInf1 - tInf0) / 1_000_000.0

                    prepMs = detInf.preprocessMs
                    quantMs = detInf.quantizeMs
                    modelMs = detInf.modelMs
                    decMs = detInf.decodeMs

                    val tStab0 = System.nanoTime()
                    val rawPred = com.echogesture.app.stabilization.RawPrediction(
                        className = detInf.predictionResult.topClassName,
                        classId = detInf.predictionResult.topClassId,
                        confidence = detInf.predictionResult.topConfidence,
                        top3 = detInf.predictionResult.top5.take(3).map { com.echogesture.app.stabilization.ClassPrediction(it) }
                    )
                    p.stabilizer.update(rawPred, true, count)
                    val tStab1 = System.nanoTime()
                    stabMs = (tStab1 - tStab0) / 1_000_000.0
                }

                val tTotal = (System.nanoTime() - tStart) / 1_000_000.0

                Log.i(
                    TAG,
                    "BENCH_STAGE: frame=$i featMs=${String.format(Locale.US, "%.3f", featMs)} " +
                    "resampleMs=${String.format(Locale.US, "%.3f", resampleMs)} " +
                    "prepMs=${String.format(Locale.US, "%.3f", prepMs)} " +
                    "quantMs=${String.format(Locale.US, "%.3f", quantMs)} " +
                    "modelMs=${String.format(Locale.US, "%.3f", modelMs)} " +
                    "decMs=${String.format(Locale.US, "%.3f", decMs)} " +
                    "infMs=${String.format(Locale.US, "%.3f", infMs)} " +
                    "stabMs=${String.format(Locale.US, "%.3f", stabMs)} " +
                    "totalMs=${String.format(Locale.US, "%.3f", tTotal)}"
                )

                try {
                    Thread.sleep(5)
                } catch (_: InterruptedException) {
                    break
                }
            }
            Log.i(TAG, "COMPLETED_BENCHMARK: frames=$targetFrames")
        }.start()
    }

    private fun initializePipeline() {
        try {
            // 1. LiteRT Model Runner
            val modelRunner = LiteRtModelRunner(this)
            val modelStatus = modelRunner.initialize(numThreads = 2)
            if (!modelStatus.isLoaded) {
                binding.tvModelStatus.text = getString(R.string.error_model_init, modelStatus.errorMessage.orEmpty())
                Log.e(TAG, "LiteRT Initialization Failed: ${modelStatus.errorMessage}")
                return
            }
            binding.tvModelStatus.text = getString(R.string.label_model_ready)

            // 2. Scaler Normalizer & Preprocessor
            val normalizer = SequenceNormalizer.fromAssets(this, "scaler_params.json")
            val preprocessor = SequencePreprocessor263D(normalizer)

            // 3. MediaPipe Hand Landmark Detector
            val detector = HandLandmarkDetector(this)

            // 4. Create Recognition Pipeline
            val recPipeline = RecognitionPipeline(
                context = this,
                detector = detector,
                modelRunner = modelRunner,
                preprocessor = preprocessor
            )
            recPipeline.stateListener = PipelineStateListener { uiState: PipelineUiState ->
                runOnUiThread {
                    updateUi(uiState)
                }
            }
            this.pipeline = recPipeline

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize recognition pipeline: ${e.message}", e)
            binding.tvModelStatus.text = getString(R.string.error_pipeline_init, e.message.orEmpty())
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                binding.permissionContainer.visibility = View.GONE
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        val controller = CameraController(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            frameListener = { imageProxy ->
                pipeline?.processFrame(
                    imageProxy = imageProxy,
                    isFrontCamera = cameraController?.isFrontCamera == true
                )
            },
            statusListener = { status ->
                runOnUiThread {
                    binding.tvCameraStatus.text = status
                }
            }
        )
        this.cameraController = controller
        controller.start()
    }

    private fun setupListeners() {
        binding.btnToggleCapture.setOnClickListener {
            isCapturing = !isCapturing
            if (isCapturing) {
                cameraController?.resume()
                binding.btnToggleCapture.text = getString(R.string.action_stop)
            } else {
                cameraController?.pause()
                binding.btnToggleCapture.text = getString(R.string.action_start)
                binding.landmarkOverlayView.clear()
                ttsManager?.stop()
            }
        }

        binding.btnSwitchCamera.setOnClickListener {
            cameraController?.toggleCamera()
            binding.landmarkOverlayView.clear()
        }

        binding.btnGrantPermission.setOnClickListener {
            checkCameraPermission()
        }

        binding.btnToggleSpeech.setOnClickListener {
            val coordinator = speechCoordinator ?: return@setOnClickListener
            val tts = ttsManager ?: return@setOnClickListener
            val newState = !coordinator.isEnabled
            coordinator.isEnabled = newState
            tts.isEnabled = newState
            binding.btnToggleSpeech.text = if (newState) {
                getString(R.string.action_speech_on)
            } else {
                getString(R.string.action_speech_off)
            }
            updateSpeechStatusUi()
        }

        binding.btnReplay.setOnClickListener {
            speechCoordinator?.replay()
            updateSpeechStatusUi()
        }

        binding.btnClear.setOnClickListener {
            speechCoordinator?.clear()
            pipeline?.temporalBuffer?.clear()
            pipeline?.stabilizer?.reset()
            binding.landmarkOverlayView.clear()
            binding.tvRecognizedSign.text = getString(R.string.label_recognized_sign)
            binding.tvConfidence.text = getString(R.string.label_confidence)
            binding.tvConfidence.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_blue))
            binding.tvTopCandidates.text = getString(R.string.label_top_candidates)
            updateSpeechStatusUi()
        }
    }

    private fun updateSpeechStatusUi() {
        val coordinator = speechCoordinator
        val tts = ttsManager

        if (coordinator == null || tts == null || !tts.isInitialized) {
            binding.tvSpeechStatus.text = getString(R.string.label_speech_unavailable)
            binding.tvSpeechStatus.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_red))
            binding.btnReplay.isEnabled = false
            return
        }

        if (!coordinator.isEnabled || !tts.isEnabled) {
            binding.tvSpeechStatus.text = getString(R.string.label_speech_muted)
            binding.tvSpeechStatus.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_amber))
        } else if (tts.isSpeaking) {
            binding.tvSpeechStatus.text = getString(R.string.label_speech_speaking)
            binding.tvSpeechStatus.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_green))
        } else {
            val lastSpoken = coordinator.lastSpokenSign
            if (lastSpoken != null) {
                binding.tvSpeechStatus.text = getString(R.string.label_last_spoken, lastSpoken.uppercase())
            } else {
                binding.tvSpeechStatus.text = getString(R.string.label_speech_ready)
            }
            binding.tvSpeechStatus.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_cyan))
        }

        binding.btnReplay.isEnabled = coordinator.lastSpokenSign != null
    }

    private fun updateUi(uiState: PipelineUiState) {
        // 1. Update Skeleton Landmark Overlay
        binding.landmarkOverlayView.setHands(uiState.detectedHands, uiState.isFrontCamera)

        // 2. Update Recognition State Label and Color
        val (stateText, stateColorRes) = when (uiState.state) {
            RecognitionState.WAITING_FOR_HANDS -> Pair(
                getString(R.string.status_waiting_for_hands),
                R.color.hud_accent_amber
            )
            RecognitionState.COLLECTING_FRAMES -> Pair(
                getString(R.string.format_collecting_frames, getString(R.string.status_collecting_frames), uiState.bufferFill),
                R.color.hud_accent_blue
            )
            RecognitionState.RECOGNIZING -> Pair(
                getString(R.string.status_recognizing),
                R.color.hud_accent_cyan
            )
            RecognitionState.ACTIVE_PREDICTION -> Pair(
                getString(R.string.status_active_prediction),
                R.color.hud_accent_green
            )
            RecognitionState.TRACKING_LOST -> Pair(
                getString(R.string.format_tracking_lost, getString(R.string.status_tracking_lost), uiState.consecutiveIdleFrames),
                R.color.hud_accent_amber
            )
            RecognitionState.IDLE -> Pair(
                getString(R.string.status_idle),
                R.color.hud_text_secondary
            )
        }
        binding.tvRecognitionState.text = stateText
        binding.tvRecognitionState.setTextColor(ContextCompat.getColor(this, stateColorRes))

        // 3. Update Hero Recognized Sign and Confidence
        if (uiState.state == RecognitionState.ACTIVE_PREDICTION ||
            (uiState.state == RecognitionState.TRACKING_LOST && uiState.recognizedSign != "IDLE")) {
            binding.tvRecognizedSign.text = uiState.recognizedSign.uppercase()
            binding.tvConfidence.text = getString(R.string.format_confidence, uiState.confidence * 100f)
            binding.tvConfidence.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_green))
        } else {
            binding.tvRecognizedSign.text = uiState.recognizedSign
            binding.tvConfidence.text = getString(R.string.format_confidence, uiState.confidence * 100f)
            binding.tvConfidence.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_blue))
        }

        // 4. Update Top Candidates Ranking
        if (uiState.topCandidates.isNotEmpty() && uiState.state != RecognitionState.IDLE && uiState.state != RecognitionState.WAITING_FOR_HANDS) {
            val candidateText = uiState.topCandidates.joinToString("  |  ") {
                "${it.className}: ${(it.confidence * 100).toInt()}%"
            }
            binding.tvTopCandidates.text = getString(R.string.format_top_candidates, candidateText)
            binding.tvTopCandidates.visibility = View.VISIBLE
        } else {
            binding.tvTopCandidates.text = getString(R.string.label_top_candidates)
        }

        // 5. Update FPS and Latency Breakdown Telemetry
        binding.tvFpsInfo.text = getString(R.string.format_fps, uiState.fps)
        binding.tvLatencyBreakdown.text = getString(
            R.string.format_latency_breakdown,
            uiState.latencies.totalMs,
            uiState.latencies.detectionMs + uiState.latencies.featureMs,
            uiState.latencies.inferenceMs
        )

        // 6. Speech Coordination & Status Update
        speechCoordinator?.onPredictionState(uiState)
        updateSpeechStatusUi()
    }

    override fun onResume() {
        super.onResume()
        previewFrameCount = 0
        lastPreviewFpsTimestamp = 0L
        Choreographer.getInstance().postFrameCallback(frameCallback)
    }

    override fun onPause() {
        super.onPause()
        Choreographer.getInstance().removeFrameCallback(frameCallback)
        ttsManager?.stop()
    }

    override fun onDestroy() {
        super.onDestroy()
        Choreographer.getInstance().removeFrameCallback(frameCallback)
        cameraController?.shutdown()
        pipeline?.close()
        speechCoordinator?.stop()
        ttsManager?.shutdown()
    }
}
