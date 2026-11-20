package com.echogesture.app.temporal

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Comprehensive unit test suite for [LiveTemporalBuffer].
 *
 * Validates:
 * 1. Rolling buffer capacity limits, FIFO eviction, and chronological ordering.
 * 2. Hand presence and consecutive idle frame tracking.
 * 3. Automatic idle reset after 12 consecutive empty frames.
 * 4. Full sequence snapshotting ([getSnapshot]) and frame retrieval.
 * 5. Temporal resampling of incomplete sequences (< 64 frames) via [TemporalResampler].
 * 6. Destination buffer reuse for zero-allocation Android runtime execution.
 * 7. Explicit [idleReset] and [clear] behavior.
 * 8. Thread safety under concurrent write and read operations.
 */
class LiveTemporalBufferTest {

    private lateinit var buffer: LiveTemporalBuffer

    companion object {
        private const val FEATURE_DIM = LiveTemporalBuffer.FEATURE_DIM // 134
        private const val INDEX_PRES_LEFT = LiveTemporalBuffer.INDEX_PRES_LEFT // 126
        private const val INDEX_PRES_RIGHT = LiveTemporalBuffer.INDEX_PRES_RIGHT // 127
    }

    @Before
    fun setUp() {
        buffer = LiveTemporalBuffer(capacity = 64, idleThreshold = 12, autoResetOnIdle = true)
    }

    private fun createFrame(tag: Float, hasLeft: Boolean = true, hasRight: Boolean = false): FloatArray {
        val frame = FloatArray(FEATURE_DIM)
        frame[0] = tag // tag to identify frame
        frame[INDEX_PRES_LEFT] = if (hasLeft) 1.0f else 0.0f
        frame[INDEX_PRES_RIGHT] = if (hasRight) 1.0f else 0.0f
        return frame
    }

    @Test
    fun testInitialState() {
        assertEquals(0, buffer.size)
        assertTrue(buffer.isEmpty)
        assertFalse(buffer.isFull)
        assertEquals(0, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)
        assertEquals(0, buffer.activeFramesCount)
        assertNull(buffer.getLatestFrame())

        val snapshot = buffer.getSnapshot()
        assertEquals(0, snapshot.size)
    }

    @Test
    fun testRollingFifoCapacityAndChronologicalOrder() {
        // Add 70 frames (capacity is 64)
        for (i in 1..70) {
            val frame = createFrame(tag = i.toFloat(), hasLeft = true)
            buffer.addFrame(frame)
        }

        assertEquals(64, buffer.size)
        assertTrue(buffer.isFull)
        assertFalse(buffer.isEmpty)
        assertEquals(70, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)

        val snapshot = buffer.getSnapshot()
        assertEquals(64, snapshot.size)

        // Oldest frame must be frame 7 (1..6 were dropped)
        assertEquals(7.0f, snapshot[0][0], 1e-6f)
        // Newest frame must be frame 70
        assertEquals(70.0f, snapshot[63][0], 1e-6f)

        // Verify strict monotonic ordering throughout
        for (j in 0 until 64) {
            val expectedTag = (j + 7).toFloat()
            assertEquals(expectedTag, snapshot[j][0], 1e-6f)
            assertEquals(expectedTag, buffer.getFrame(j)[0], 1e-6f)
        }

        val latest = buffer.getLatestFrame()
        assertNotNull(latest)
        assertEquals(70.0f, latest!![0], 1e-6f)
    }

    @Test
    fun testPresenceTrackingAndIdleCounters() {
        // 5 presence frames
        for (i in 1..5) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = true))
        }
        assertEquals(5, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)
        assertEquals(5, buffer.activeFramesCount)

        // 3 idle frames (no hands)
        for (i in 6..8) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = false, hasRight = false))
        }
        assertEquals(0, buffer.consecutivePresenceFrames)
        assertEquals(3, buffer.consecutiveIdleFrames)
        assertEquals(5, buffer.activeFramesCount) // 5 of 8 have active hands
        assertEquals(8, buffer.size)

        // 1 hand frame reappears
        buffer.addFrame(createFrame(tag = 9.0f, hasLeft = false, hasRight = true))
        assertEquals(1, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)
        assertEquals(6, buffer.activeFramesCount)
    }

    @Test
    fun testAutomaticIdleResetAfter12EmptyFrames() {
        // Fill 20 frames with hands
        for (i in 1..20) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = true))
        }
        assertEquals(20, buffer.size)
        assertEquals(20, buffer.consecutivePresenceFrames)

        // Add 11 idle frames (idleThreshold is 12)
        for (i in 1..11) {
            val hasHands = buffer.addFrame(createFrame(tag = 100f + i, hasLeft = false, hasRight = false))
            assertFalse(hasHands)
            assertEquals(i, buffer.consecutiveIdleFrames)
            assertEquals(0, buffer.consecutivePresenceFrames)
        }
        // Buffer should still contain 31 frames (20 + 11)
        assertEquals(31, buffer.size)

        // The 12th idle frame triggers autoResetOnIdle
        val hasHands12 = buffer.addFrame(createFrame(tag = 112f, hasLeft = false, hasRight = false))
        assertFalse(hasHands12)
        assertEquals(12, buffer.consecutiveIdleFrames)
        assertEquals(0, buffer.consecutivePresenceFrames)

        // Buffer must now be cleared!
        assertEquals(0, buffer.size)
        assertTrue(buffer.isEmpty)

        // Adding 13th idle frame keeps buffer clear and increments idle count
        buffer.addFrame(createFrame(tag = 113f, hasLeft = false, hasRight = false))
        assertEquals(13, buffer.consecutiveIdleFrames)
        assertEquals(0, buffer.size)
    }

    @Test
    fun testExplicitIdleResetAndClear() {
        for (i in 1..10) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = true))
        }
        assertEquals(10, buffer.size)
        assertEquals(10, buffer.consecutivePresenceFrames)

        // Explicit idleReset clears buffer and presence, keeps idle
        buffer.idleReset()
        assertEquals(0, buffer.size)
        assertEquals(0, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)

        // Add 3 idle frames
        for (i in 1..3) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = false))
        }
        assertEquals(3, buffer.consecutiveIdleFrames)

        // Explicit clear zeroes idle counter as well
        buffer.clear()
        assertEquals(0, buffer.size)
        assertEquals(0, buffer.consecutivePresenceFrames)
        assertEquals(0, buffer.consecutiveIdleFrames)
    }

    @Test
    fun testIncompleteSequenceResampling16Frames() {
        // Simulate a 16-frame gesture sequence
        for (i in 0 until 16) {
            val frame = createFrame(tag = i.toFloat(), hasLeft = true)
            // Left wrist x landmark at index 0, y at index 1
            frame[0] = 0.2f + i * 0.02f
            frame[1] = 0.5f + i * 0.01f
            buffer.addFrame(frame)
        }
        assertEquals(16, buffer.size)

        // Resample 16 frames to target 64 frames
        val resampled = buffer.getResampledSnapshot(targetLength = 64)
        assertEquals(64, resampled.size)

        // Validate presence is preserved
        for (j in 0 until 64) {
            assertEquals(1.0f, resampled[j][INDEX_PRES_LEFT], 1e-6f)
            assertEquals(0.0f, resampled[j][INDEX_PRES_RIGHT], 1e-6f)
        }

        // Validate linear interpolation endpoint alignment
        assertEquals(0.2f, resampled[0][0], 1e-5f)
        val expectedEndTag = 0.2f + 15 * 0.02f // 0.50f
        assertEquals(expectedEndTag, resampled[63][0], 1e-5f)

        // Middle frame should be smoothly interpolated
        assertTrue(resampled[32][0] > resampled[0][0])
        assertTrue(resampled[32][0] < resampled[63][0])
    }

    @Test
    fun testResampleEmptyBufferReturnsZeros() {
        val resampled = buffer.getResampledSnapshot(targetLength = 64)
        assertEquals(64, resampled.size)
        for (j in 0 until 64) {
            assertEquals(FEATURE_DIM, resampled[j].size)
            for (d in 0 until FEATURE_DIM) {
                assertEquals(0.0f, resampled[j][d], 1e-7f)
            }
        }
    }

    @Test
    fun testResampleWithDestinationBuffer() {
        for (i in 0 until 32) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = true))
        }

        val dest = Array(64) { FloatArray(FEATURE_DIM) }
        val returned = buffer.getResampledSnapshot(targetLength = 64, destination = dest)

        // Must write directly into dest instance
        assertTrue(dest === returned)
        assertEquals(64, dest.size)
        assertEquals(1.0f, dest[0][INDEX_PRES_LEFT], 1e-6f)
    }

    @Test
    fun testFullCapacitySnapshotDirectCopy() {
        for (i in 0 until 64) {
            buffer.addFrame(createFrame(tag = i.toFloat(), hasLeft = true))
        }
        assertEquals(64, buffer.size)

        val dest = Array(64) { FloatArray(FEATURE_DIM) }
        val result = buffer.getResampledSnapshot(targetLength = 64, destination = dest)
        assertTrue(dest === result)
        assertEquals(0.0f, result[0][0], 1e-6f)
        assertEquals(63.0f, result[63][0], 1e-6f)
    }

    @Test
    fun testThreadSafetyConcurrentReadWrite() {
        val threadCount = 8
        val framesPerThread = 200
        val executor = Executors.newFixedThreadPool(threadCount)
        val startLatch = CountDownLatch(1)
        val doneLatch = CountDownLatch(threadCount)
        val writeCounter = AtomicInteger(0)

        for (t in 0 until threadCount) {
            executor.submit {
                startLatch.await()
                try {
                    for (i in 0 until framesPerThread) {
                        val id = writeCounter.incrementAndGet()
                        buffer.addFrame(createFrame(tag = id.toFloat(), hasLeft = (id % 2 == 0)))
                        if (i % 10 == 0) {
                            val snap = buffer.getSnapshot()
                            assertTrue(snap.size <= 64)
                        }
                    }
                } finally {
                    doneLatch.countDown()
                }
            }
        }

        startLatch.countDown()
        assertTrue("Concurrent test timed out", doneLatch.await(10, TimeUnit.SECONDS))
        executor.shutdown()

        assertEquals(64, buffer.size)
        val finalSnapshot = buffer.getSnapshot()
        assertEquals(64, finalSnapshot.size)
    }
}
