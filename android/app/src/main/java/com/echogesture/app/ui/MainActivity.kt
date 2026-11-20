package com.echogesture.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
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
import com.echogesture.app.stabilization.RecognitionState
import com.echogesture.app.vision.HandLandmarkDetector

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private var cameraController: CameraController? = null
    private var pipeline: RecognitionPipeline? = null
    private var isCapturing = true

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
        setupListeners()
        checkCameraPermission()
    }

    private fun initializePipeline() {
        try {
            // 1. LiteRT Model Runner
            val modelRunner = LiteRtModelRunner(this)
            val modelStatus = modelRunner.initialize(numThreads = 2)
            if (!modelStatus.isLoaded) {
                binding.tvModelStatus.text = "Model Error: ${modelStatus.errorMessage}"
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
            binding.tvModelStatus.text = "Init Error: ${e.message}"
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
            }
        }

        binding.btnSwitchCamera.setOnClickListener {
            cameraController?.toggleCamera()
            binding.landmarkOverlayView.clear()
        }

        binding.btnGrantPermission.setOnClickListener {
            checkCameraPermission()
        }
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
                "${getString(R.string.status_collecting_frames)} (${uiState.bufferFill}/16)",
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
                "${getString(R.string.status_tracking_lost)} (${uiState.consecutiveIdleFrames}/12)",
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
            binding.tvConfidence.text = String.format("Confidence: %.1f%%", uiState.confidence * 100f)
            binding.tvConfidence.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_green))
        } else {
            binding.tvRecognizedSign.text = uiState.recognizedSign
            binding.tvConfidence.text = String.format("Confidence: %.1f%%", uiState.confidence * 100f)
            binding.tvConfidence.setTextColor(ContextCompat.getColor(this, R.color.hud_accent_blue))
        }

        // 4. Update Top Candidates Ranking
        if (uiState.topCandidates.isNotEmpty() && uiState.state != RecognitionState.IDLE && uiState.state != RecognitionState.WAITING_FOR_HANDS) {
            val candidateText = uiState.topCandidates.joinToString("  |  ") {
                "${it.className}: ${(it.confidence * 100).toInt()}%"
            }
            binding.tvTopCandidates.text = "Top: $candidateText"
            binding.tvTopCandidates.visibility = View.VISIBLE
        } else {
            binding.tvTopCandidates.text = getString(R.string.label_top_candidates)
        }

        // 5. Update FPS and Latency Breakdown Telemetry
        binding.tvFpsInfo.text = String.format("FPS: %.1f", uiState.fps)
        binding.tvLatencyBreakdown.text = String.format(
            "Latency: %.1f ms (Vision: %.1f ms | Inf: %.1f ms)",
            uiState.latencies.totalMs,
            uiState.latencies.detectionMs + uiState.latencies.featureMs,
            uiState.latencies.inferenceMs
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraController?.shutdown()
        pipeline?.close()
    }
}
