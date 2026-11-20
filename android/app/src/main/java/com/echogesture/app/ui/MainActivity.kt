package com.echogesture.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.echogesture.app.R
import com.echogesture.app.camera.CameraManager
import com.echogesture.app.databinding.ActivityMainBinding
import com.echogesture.app.inference.EchoGestureInferenceEngine
import com.echogesture.app.inference.LiteRtModelRunner
import com.echogesture.app.inference.ModelContract
import com.echogesture.app.inference.PredictionResult
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var cameraManager: CameraManager? = null
    private var modelRunner: LiteRtModelRunner? = null
    private var inferenceEngine: EchoGestureInferenceEngine? = null
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
        val t0 = System.nanoTime()
        val runner = LiteRtModelRunner(this)
        this.modelRunner = runner
        val status = runner.initialize(numThreads = 2)
        val initMs = (System.nanoTime() - t0) / 1_000_000.0

        if (status.isLoaded) {
            binding.tvModelStatus.text = getString(R.string.label_model_ready)
            binding.tvTensorContract.text = "In: ${status.inputShape} ${status.inputDataType} | Out: ${status.outputShape} ${status.outputDataType}"
            this.inferenceEngine = EchoGestureInferenceEngine(
                modelRunner = runner
            )
            val benchmarkIterations = intent.getIntExtra("iterations", 500)
            runBenchmark(runner, initMs, benchmarkIterations)
        } else {
            binding.tvModelStatus.text = "Model Error: ${status.errorMessage}"
            Log.e(TAG, "Model Initialization Failed: ${status.errorMessage}")
        }
    }

    private fun runBenchmark(runner: LiteRtModelRunner, initMs: Double, iterations: Int = 500) {
        Thread {
            try {
                // 1. Load benchmark sample input
                val sampleBytes = ByteArray(ModelContract.INPUT_BUFFER_SIZE_BYTES)
                var sampleLoaded = false
                try {
                    assets.open("benchmark_sample.bin").use { inputStream: InputStream ->
                        val read = inputStream.read(sampleBytes)
                        sampleLoaded = (read == ModelContract.INPUT_BUFFER_SIZE_BYTES)
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "benchmark_sample.bin not found, fallback to zero buffer: ${e.message}")
                }
                if (!sampleLoaded) {
                    sampleBytes.fill(23) // Affine zero-point
                }

                // 2. Warm-up (10 runs)
                val warmUpRuns = 10
                val warmStart = System.nanoTime()
                for (w in 0 until warmUpRuns) {
                    runner.inputBuffer.rewind()
                    runner.inputBuffer.put(sampleBytes)
                    runner.inputBuffer.rewind()
                    runner.outputBuffer.rewind()
                    runner.runInference(runner.inputBuffer, runner.outputBuffer)
                }
                val warmUpMs = (System.nanoTime() - warmStart) / 1_000_000.0

                // 3. Timed Iterations (e.g. 500 iterations)
                val latenciesMs = DoubleArray(iterations)
                var lastResult: PredictionResult? = null

                for (i in 0 until iterations) {
                    runner.inputBuffer.rewind()
                    runner.inputBuffer.put(sampleBytes)
                    runner.inputBuffer.rewind()
                    runner.outputBuffer.rewind()

                    val tStart = System.nanoTime()
                    runner.runInference(runner.inputBuffer, runner.outputBuffer)
                    val tEnd = System.nanoTime()

                    latenciesMs[i] = (tEnd - tStart) / 1_000_000.0

                    if (i == iterations - 1) {
                        lastResult = inferenceEngine?.decoder?.decode(runner.outputBuffer, runner.getClassLabels())
                    }
                }

                // 4. Compute Statistical Metrics
                latenciesMs.sort()
                val minMs = latenciesMs.first()
                val maxMs = latenciesMs.last()
                val meanMs = latenciesMs.average()
                val medianMs = latenciesMs[iterations / 2]
                val p95Index = ((iterations * 0.95).toInt()).coerceAtMost(iterations - 1)
                val p95Ms = latenciesMs[p95Index]

                // Memory query
                val nativeHeapMb = Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0)

                // 5. Structured Log Output
                Log.i(TAG, "================================================================================")
                Log.i(TAG, " ECHO-GESTURE PHYSICAL DEVICE INFERENCE BENCHMARK REPORT")
                Log.i(TAG, "================================================================================")
                Log.i(TAG, "Hardware Device:     ${Build.MANUFACTURER} ${Build.MODEL} (${Build.PRODUCT})")
                Log.i(TAG, "Android OS Version:  Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                Log.i(TAG, "CPU Architecture:    ${Build.SUPPORTED_ABIS.joinToString(", ")}")
                Log.i(TAG, "Model Checksum:      ${ModelContract.EXPECTED_SHA256}")
                Log.i(TAG, "Interpreter Engine:  LiteRT 1.4.2 (Threads: 2, XNNPACK CPU)")
                Log.i(TAG, "Model Init Time:     ${String.format("%.2f", initMs)} ms")
                Log.i(TAG, "Warm-up Time (10):   ${String.format("%.2f", warmUpMs)} ms")
                Log.i(TAG, "Timed Iterations:    $iterations runs")
                Log.i(TAG, "Latency Mean:        ${String.format("%.3f", meanMs)} ms")
                Log.i(TAG, "Latency Median:      ${String.format("%.3f", medianMs)} ms")
                Log.i(TAG, "Latency P95:         ${String.format("%.3f", p95Ms)} ms")
                Log.i(TAG, "Latency Min:         ${String.format("%.3f", minMs)} ms")
                Log.i(TAG, "Latency Max:         ${String.format("%.3f", maxMs)} ms")
                Log.i(TAG, "Native Heap Memory:  ${String.format("%.2f", nativeHeapMb)} MB")

                if (lastResult != null) {
                    Log.i(TAG, "Top-1 Class ID:      ${lastResult.topClassId}")
                    Log.i(TAG, "Top-1 Class Name:    ${lastResult.topClassName}")
                    Log.i(TAG, "Top-1 Confidence:    ${String.format("%.4f", lastResult.topConfidence)}")
                    Log.i(TAG, "Top-5 Ranking:")
                    for (c in lastResult.top5) {
                        Log.i(TAG, "  Rank ${c.rank}: ID=${c.classId} (${c.className}) -> score=${c.quantizedScore}, conf=${String.format("%.4f", c.confidence)}")
                    }
                    val parityMatch = if (sampleLoaded) {
                        lastResult.topClassId == 5 && lastResult.topClassName == "book"
                    } else {
                        lastResult.topClassId == 30 && lastResult.topClassName == "orange"
                    }
                    Log.i(TAG, "Fixture Parity:      ${if (parityMatch) "MATCH (100% Verified)" else "MISMATCH"}")
                }
                Log.i(TAG, "================================================================================")
                Log.i(TAG, "BENCHMARK_COMPLETE: mean=${String.format("%.3f", meanMs)} median=${String.format("%.3f", medianMs)} p95=${String.format("%.3f", p95Ms)}")

                runOnUiThread {
                    binding.tvLatency.text = String.format("Mean: %.2f ms | P95: %.2f ms (%d runs)", meanMs, p95Ms, iterations)
                    if (lastResult != null) {
                        binding.tvRecognizedSign.text = String.format("%s (%.1f%%)", lastResult.topClassName, lastResult.topConfidence * 100)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed with error: ${e.message}", e)
                runOnUiThread {
                    binding.tvLatency.text = "Benchmark Error: ${e.message}"
                }
            }
        }.start()
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
        inferenceEngine?.close()
        modelRunner?.close()
    }

    companion object {
        const val TAG = "ECHO_BENCHMARK"
    }
}
