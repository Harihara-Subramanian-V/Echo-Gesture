package com.echogesture.app.inference

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

data class ModelStatus(
    val isLoaded: Boolean,
    val inputShape: List<Int>,
    val outputShape: List<Int>,
    val inputDataType: String,
    val outputDataType: String,
    val classCount: Int,
    val errorMessage: String? = null
)

class LiteRtModelRunner(private val context: Context) : AutoCloseable {

    private var interpreter: Interpreter? = null
    private val classLabels = mutableListOf<String>()

    val inputBuffer: ByteBuffer = ModelContract.createDirectByteBuffer(ModelContract.INPUT_BUFFER_SIZE_BYTES)
    val outputBuffer: ByteBuffer = ModelContract.createDirectByteBuffer(ModelContract.OUTPUT_BUFFER_SIZE_BYTES)

    fun initialize(): ModelStatus {
        return try {
            val modelBuffer = loadModelFile(ModelContract.MODEL_ASSET_PATH)
            val options = Interpreter.Options().apply {
                setNumThreads(1)
            }
            val interp = Interpreter(modelBuffer, options)
            this.interpreter = interp

            val inTensor = interp.getInputTensor(0)
            val outTensor = interp.getOutputTensor(0)

            val inShape = inTensor.shape().toList()
            val outShape = outTensor.shape().toList()

            check(inTensor.dataType() == DataType.INT8) { "Input tensor must be INT8" }
            check(outTensor.dataType() == DataType.INT8) { "Output tensor must be INT8" }
            check(inShape == listOf(1, 64, 263)) { "Expected input shape [1, 64, 263], got $inShape" }
            check(outShape == listOf(1, 40)) { "Expected output shape [1, 40], got $outShape" }

            loadVocabulary(ModelContract.LABEL_ENCODER_PATH)

            ModelStatus(
                isLoaded = true,
                inputShape = inShape,
                outputShape = outShape,
                inputDataType = inTensor.dataType().name,
                outputDataType = outTensor.dataType().name,
                classCount = classLabels.size
            )
        } catch (e: Exception) {
            ModelStatus(
                isLoaded = false,
                inputShape = emptyList(),
                outputShape = emptyList(),
                inputDataType = "UNKNOWN",
                outputDataType = "UNKNOWN",
                classCount = 0,
                errorMessage = e.message
            )
        }
    }

    fun runInference(): Pair<Int, Float> {
        val interp = interpreter ?: throw IllegalStateException("Model not initialized")
        inputBuffer.rewind()
        outputBuffer.rewind()
        interp.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val outBytes = ByteArray(ModelContract.NUM_CLASSES)
        outputBuffer.get(outBytes)

        val topIdx = ModelContract.computeArgMax(outBytes)
        val topConf = if (topIdx in 0 until ModelContract.NUM_CLASSES) {
            ModelContract.dequantize(outBytes[topIdx])
        } else {
            0.0f
        }
        return Pair(topIdx, topConf)
    }

    fun getClassName(classIdx: Int): String {
        return if (classIdx in 0 until classLabels.size) classLabels[classIdx] else "UNKNOWN"
    }

    private fun loadModelFile(assetPath: String): ByteBuffer {
        val afd = context.assets.openFd(assetPath)
        FileInputStream(afd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        }
    }

    private fun loadVocabulary(assetPath: String) {
        classLabels.clear()
        context.assets.open(assetPath).bufferedReader().use { reader ->
            val json = JSONObject(reader.readText())
            val idx2label = json.getJSONObject("idx2label")
            for (i in 0 until idx2label.length()) {
                classLabels.add(idx2label.getString(i.toString()))
            }
        }
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
    }
}
