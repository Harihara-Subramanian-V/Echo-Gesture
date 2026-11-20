package com.echogesture.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.echogesture.app.R
import com.echogesture.app.camera.CameraManager
import com.echogesture.app.databinding.ActivityMainBinding
import com.echogesture.app.inference.LiteRtModelRunner
import com.echogesture.app.inference.ModelContract

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var cameraManager: CameraManager? = null
    private var modelRunner: LiteRtModelRunner? = null
    private var isCapturing = true

    private var frameCount = 0
    private var lastFpsTimestamp = System.currentTimeMillis()
    private var currentFps = 0.0

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

        initializeModel()
        setupListeners()
        checkCameraPermission()
    }

    private fun initializeModel() {
        val runner = LiteRtModelRunner(this)
        this.modelRunner = runner
        val status = runner.initialize()

        if (status.isLoaded) {
            binding.tvModelStatus.text = getString(R.string.label_model_ready)
            binding.tvTensorContract.text = "In: ${status.inputShape} ${status.inputDataType} | Out: ${status.outputShape} ${status.outputDataType}"
            benchmarkModel(runner)
        } else {
            binding.tvModelStatus.text = "Model Error: ${status.errorMessage}"
        }
    }

    private fun benchmarkModel(runner: LiteRtModelRunner) {
        try {
            // Warmup & timing test with zeros
            val t0 = System.nanoTime()
            runner.runInference()
            val t1 = System.nanoTime()
            val invokeMs = (t1 - t0) / 1_000_000.0
            binding.tvLatency.text = String.format("Invoke Latency: %.3f ms (XNNPACK CPU)", invokeMs)
        } catch (e: Exception) {
            binding.tvLatency.text = "Benchmark: ${e.message}"
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED -> {
                binding.permissionContainer.visibility = View.GONE
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        cameraManager = CameraManager(
            context = this,
            lifecycleOwner = this,
            previewView = binding.previewView,
            onFrameReceived = { imageProxy ->
                frameCount++
                val now = System.currentTimeMillis()
                if (now - lastFpsTimestamp >= 1000) {
                    currentFps = frameCount * 1000.0 / (now - lastFpsTimestamp)
                    frameCount = 0
                    lastFpsTimestamp = now
                    runOnUiThread {
                        binding.tvCameraStatus.text = String.format(
                            "Camera: %dx%d @ %.1f FPS",
                            imageProxy.width, imageProxy.height, currentFps
                        )
                    }
                }
            },
            onStatusUpdate = { status ->
                runOnUiThread {
                    binding.tvCameraStatus.text = status
                }
            }
        ).also { it.start() }
    }

    private fun setupListeners() {
        binding.btnGrantPermission.setOnClickListener {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        binding.btnToggleCapture.setOnClickListener {
            if (isCapturing) {
                cameraManager?.pause()
                binding.btnToggleCapture.text = getString(R.string.action_start)
                binding.tvRecognizedSign.text = getString(R.string.label_recognized_sign)
            } else {
                cameraManager?.resume()
                binding.btnToggleCapture.text = getString(R.string.action_stop)
            }
            isCapturing = !isCapturing
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraManager?.release()
        modelRunner?.close()
    }
}
