package com.echogesture.app.inference

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Host JVM unit test suite for LiteRtModelRunner.
 *
 * Validates:
 * - Model file loading from assets (via file and direct buffer)
 * - Strict ModelContract tensor shape, datatype, and affine quantization parameters
 * - Complete 40-class vocabulary verification
 * - Zero-allocation DirectByteBuffer specifications
 * - Inference execution across ByteBuffer, parameterless, and ByteArray APIs
 * - Dummy all-zero (byte = 23) inference behavior
 * - Robust error handling, thread configuration, and AutoCloseable lifecycle safety
 */
class LiteRtModelRunnerTest {

    private fun findAssetFile(fileName: String): File {
        val candidates = listOf(
            File("src/main/assets", fileName),
            File("app/src/main/assets", fileName),
            File("../app/src/main/assets", fileName),
            File("../../app/src/main/assets", fileName),
            File("android/app/src/main/assets", fileName)
        )
        val file = candidates.firstOrNull { it.exists() }
        assertNotNull("Could not find asset file '$fileName' in candidate locations", file)
        return file!!
    }

    @Test
    fun testInitializeFromFileAndValidateContract() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromFile(modelFile, vocabFile, numThreads = 2)

            assertTrue("Model should be successfully loaded", status.isLoaded)
            assertTrue("Runner should report loaded", runner.isLoaded)
            assertNull("Error message should be null on success", status.errorMessage)

            // Validate Input Tensor: [1, 64, 263], INT8, S ~0.057940476, Z = 23
            assertEquals(listOf(1, 64, 263), status.inputShape)
            assertEquals(listOf(1, 64, 263), runner.inputShape)
            assertEquals("INT8", status.inputDataType)
            assertEquals("INT8", runner.inputDataType)
            assertEquals(ModelContract.INPUT_SCALE, status.inputScale, 1e-4f)
            assertEquals(ModelContract.INPUT_SCALE, runner.inputScale, 1e-4f)
            assertEquals(ModelContract.INPUT_ZERO_POINT, status.inputZeroPoint)
            assertEquals(ModelContract.INPUT_ZERO_POINT, runner.inputZeroPoint)

            // Validate Output Tensor: [1, 40], INT8, S ~0.00390625, Z = -128
            assertEquals(listOf(1, 40), status.outputShape)
            assertEquals(listOf(1, 40), runner.outputShape)
            assertEquals("INT8", status.outputDataType)
            assertEquals("INT8", runner.outputDataType)
            assertEquals(ModelContract.OUTPUT_SCALE, status.outputScale, 1e-5f)
            assertEquals(ModelContract.OUTPUT_SCALE, runner.outputScale, 1e-5f)
            assertEquals(ModelContract.OUTPUT_ZERO_POINT, status.outputZeroPoint)
            assertEquals(ModelContract.OUTPUT_ZERO_POINT, runner.outputZeroPoint)

            // Validate Vocabulary
            assertEquals(40, status.classCount)
            val labels = runner.getClassLabels()
            assertEquals(40, labels.size)
            assertEquals("all", labels[0])
            assertEquals("yes", labels[39])
            assertEquals("all", runner.getClassName(0))
            assertEquals("yes", runner.getClassName(39))
            assertEquals("UNKNOWN", runner.getClassName(-1))
            assertEquals("UNKNOWN", runner.getClassName(40))
        }
    }

    @Test
    fun testInitializeFromBuffer() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)

        val modelBuffer: ByteBuffer = FileInputStream(modelFile).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length())
        }
        val vocabJson = vocabFile.readText()

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromBuffer(modelBuffer, vocabJson, numThreads = 4)

            assertTrue("Model should load from buffer", status.isLoaded)
            assertEquals(listOf(1, 64, 263), status.inputShape)
            assertEquals(listOf(1, 40), status.outputShape)
            assertEquals(40, status.classCount)
        }
    }

    @Test
    fun testInferenceWithDummyAllZeroBuffer() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromFile(modelFile, vocabFile)
            assertTrue(status.isLoaded)

            // 1. Dummy all-zero buffer test:
            // Input represents float 0.0, which quantizes to signed INT8 byte = 23 (ModelContract.INPUT_ZERO_POINT)
            val dummyInput = ByteArray(ModelContract.INPUT_BUFFER_SIZE_BYTES) { ModelContract.INPUT_ZERO_POINT.toByte() }
            val outputBytes = runner.runInference(dummyInput)

            assertNotNull(outputBytes)
            assertEquals("Output byte array must have size 40", 40, outputBytes.size)

            // Verify argmax on output bytes
            val topIdx = ModelContract.computeArgMax(outputBytes)
            assertTrue("Top index should be valid class [0..39]", topIdx in 0 until 40)
            val topConf = ModelContract.dequantize(outputBytes[topIdx])
            assertTrue("Top confidence should be in valid probability range [0.0, 1.0]", topConf in 0.0f..1.0f)

            // 2. Direct ByteBuffer inference test:
            runner.inputBuffer.rewind()
            for (i in 0 until ModelContract.INPUT_BUFFER_SIZE_BYTES) {
                runner.inputBuffer.put(ModelContract.INPUT_ZERO_POINT.toByte())
            }
            runner.inputBuffer.rewind()
            runner.outputBuffer.rewind()

            runner.runInference(runner.inputBuffer, runner.outputBuffer)
            assertEquals(ModelContract.OUTPUT_BUFFER_SIZE_BYTES, runner.outputBuffer.position())

            // 3. Parameterless inference test (internal buffers):
            runner.inputBuffer.rewind()
            for (i in 0 until ModelContract.INPUT_BUFFER_SIZE_BYTES) {
                runner.inputBuffer.put(ModelContract.INPUT_ZERO_POINT.toByte())
            }
            val (paramlessTopIdx, paramlessConf) = runner.runInference()
            assertTrue(paramlessTopIdx in 0 until 40)
            assertTrue(paramlessConf in 0.0f..1.0f)
        }
    }

    @Test
    fun testDirectByteBufferProperties() {
        LiteRtModelRunner().use { runner ->
            // Invariants: native order and direct allocation for JNI zero-overhead execution
            assertTrue("Input buffer must be direct", runner.inputBuffer.isDirect)
            assertEquals(ModelContract.INPUT_BUFFER_SIZE_BYTES, runner.inputBuffer.capacity())
            assertEquals(ByteOrder.nativeOrder(), runner.inputBuffer.order())

            assertTrue("Output buffer must be direct", runner.outputBuffer.isDirect)
            assertEquals(ModelContract.OUTPUT_BUFFER_SIZE_BYTES, runner.outputBuffer.capacity())
            assertEquals(ByteOrder.nativeOrder(), runner.outputBuffer.order())
        }
    }

    @Test
    fun testLifecycleAndSafeClosing() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)

        val runner = LiteRtModelRunner()
        val status = runner.initializeFromFile(modelFile, vocabFile)
        assertTrue(status.isLoaded)
        assertTrue(runner.isLoaded)

        // Close runner
        runner.close()
        assertFalse(runner.isLoaded)

        // Calling close again should be idempotent and safe
        runner.close()
        assertFalse(runner.isLoaded)

        // Inference after close must throw IllegalStateException
        val dummyInput = ByteArray(ModelContract.INPUT_BUFFER_SIZE_BYTES) { 23.toByte() }
        assertThrows(IllegalStateException::class.java) {
            runner.runInference(dummyInput)
        }
        assertThrows(IllegalStateException::class.java) {
            runner.runInference()
        }
        assertThrows(IllegalStateException::class.java) {
            runner.runInference(runner.inputBuffer, runner.outputBuffer)
        }
    }

    @Test
    fun testUninitializedInferenceGuards() {
        LiteRtModelRunner().use { runner ->
            assertFalse(runner.isLoaded)
            val dummyInput = ByteArray(ModelContract.INPUT_BUFFER_SIZE_BYTES) { 23.toByte() }

            assertThrows(IllegalStateException::class.java) {
                runner.runInference(dummyInput)
            }
            assertThrows(IllegalStateException::class.java) {
                runner.runInference()
            }
            assertThrows(IllegalStateException::class.java) {
                runner.runInference(runner.inputBuffer, runner.outputBuffer)
            }
        }
    }

    @Test
    fun testErrorHandlingMissingFiles() {
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)
        val nonExistentModel = File("does_not_exist.tflite")

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromFile(nonExistentModel, vocabFile)
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
            assertTrue(status.errorMessage!!.contains("does not exist"))
        }

        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val nonExistentVocab = File("does_not_exist.json")

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromFile(modelFile, nonExistentVocab)
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
            assertTrue(status.errorMessage!!.contains("does not exist"))
        }
    }

    @Test
    fun testErrorHandlingInvalidBuffer() {
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)
        val vocabJson = vocabFile.readText()

        // Create buffer with invalid magic (not TFL3)
        val invalidBuffer = ByteBuffer.allocateDirect(1024)
        invalidBuffer.put("CORRUPT_BUFFER_HEADER".toByteArray())
        invalidBuffer.rewind()

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromBuffer(invalidBuffer, vocabJson)
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
        }
    }

    @Test
    fun testErrorHandlingInvalidVocabulary() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val modelBuffer: ByteBuffer = FileInputStream(modelFile).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length())
        }

        // Test 1: JSON without idx2label or classes
        val invalidJson1 = JSONObject().put("foo", "bar").toString()
        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromBuffer(modelBuffer, invalidJson1)
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
        }

        // Test 2: JSON with incorrect class count (e.g. 5 instead of 40)
        val invalidJson2 = JSONObject().apply {
            val idx2label = JSONObject()
            for (i in 0 until 5) {
                idx2label.put(i.toString(), "class_$i")
            }
            put("idx2label", idx2label)
        }.toString()

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromBuffer(modelBuffer, invalidJson2)
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
            assertTrue(status.errorMessage!!.contains("40"))
        }
    }

    @Test
    fun testAlternativeVocabularyFormatWithClassesKey() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val modelBuffer: ByteBuffer = FileInputStream(modelFile).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length())
        }

        // Vocabulary supplied via "classes" array containing exactly 40 items
        val classesJson = JSONObject().apply {
            val arr = JSONArray()
            for (i in 0 until 40) {
                arr.put("sign_$i")
            }
            put("classes", arr)
        }.toString()

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromBuffer(modelBuffer, classesJson)
            assertTrue(status.isLoaded)
            assertEquals(40, status.classCount)
            assertEquals("sign_0", runner.getClassName(0))
            assertEquals("sign_39", runner.getClassName(39))
        }
    }

    @Test
    fun testInitializeDefaultWithoutContextFailsGracefully() {
        LiteRtModelRunner(null).use { runner ->
            val status = runner.initialize()
            assertFalse(status.isLoaded)
            assertNotNull(status.errorMessage)
            assertTrue(status.errorMessage!!.contains("Context is null"))
        }
    }

    @Test
    fun testInferenceInputSizeValidation() {
        val modelFile = findAssetFile(ModelContract.MODEL_ASSET_PATH)
        val vocabFile = findAssetFile(ModelContract.LABEL_ENCODER_PATH)

        LiteRtModelRunner().use { runner ->
            val status = runner.initializeFromFile(modelFile, vocabFile)
            assertTrue(status.isLoaded)

            // Wrong size array (e.g. 100 bytes instead of 16,832)
            val wrongSizeInput = ByteArray(100)
            assertThrows(IllegalArgumentException::class.java) {
                runner.runInference(wrongSizeInput)
            }
        }
    }
}
