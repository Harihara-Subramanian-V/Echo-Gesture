package com.echogesture.app.temporal

import com.echogesture.app.preprocessing.TemporalResampler
import java.util.ArrayDeque

/**
 * Thread-safe rolling temporal buffer for live 134-D gesture frames in EchoGesture Phase 7C.
 *
 * Capabilities:
 * - Thread-safe rolling FIFO window of 134-D frames (default capacity: 64).
 * - Preserves strict chronological order (oldest at index 0, newest at index size - 1).
 * - Tracks consecutive hand presence frames and consecutive idle frames.
 * - When consecutive idle frames reach or exceed [idleThreshold] (default: 12 frames),
 *   automatically resets or supports explicit [idleReset].
 * - Supports full sequence snapshots via [getSnapshot].
 * - Supports incomplete sequence handling via [getResampledSnapshot] using [TemporalResampler.resample].
 * - Zero-allocation destination buffer support for real-time mobile efficiency.
 * - Complete state clearing via [clear].
 *
 * Vector Layout (134-D):
 * - [0:63]   : Left Hand 21 3D landmarks
 * - [63:126] : Right Hand 21 3D landmarks
 * - [126]    : Left Hand Presence flag (1.0f or 0.0f)
 * - [127]    : Right Hand Presence flag (1.0f or 0.0f)
 * - [128:129]: Left and Right Hand confidence scores
 * - [130]    : Both Hands Co-presence flag
 * - [131:133]: Inter-wrist distance and offsets
 */
class LiveTemporalBuffer(
    val capacity: Int = DEFAULT_CAPACITY,
    val idleThreshold: Int = DEFAULT_IDLE_THRESHOLD,
    val autoResetOnIdle: Boolean = true
) {
    companion object {
        const val DEFAULT_CAPACITY = 64
        const val DEFAULT_IDLE_THRESHOLD = 12
        const val FEATURE_DIM = TemporalResampler.FEATURE_DIM // 134
        const val INDEX_PRES_LEFT = TemporalResampler.INDEX_PRES_LEFT // 126
        const val INDEX_PRES_RIGHT = TemporalResampler.INDEX_PRES_RIGHT // 127
    }

    private val lock = Any()
    private val buffer = ArrayDeque<FloatArray>(capacity)

    @Volatile
    private var _consecutivePresenceFrames: Int = 0

    @Volatile
    private var _consecutiveIdleFrames: Int = 0

    /**
     * Number of consecutive frames where at least one hand was detected.
     */
    val consecutivePresenceFrames: Int
        get() = synchronized(lock) { _consecutivePresenceFrames }

    /**
     * Number of consecutive frames with no hands detected.
     */
    val consecutiveIdleFrames: Int
        get() = synchronized(lock) { _consecutiveIdleFrames }

    /**
     * Current number of frames stored in the rolling buffer.
     */
    val size: Int
        get() = synchronized(lock) { buffer.size }

    /**
     * True if buffer has no frames.
     */
    val isEmpty: Boolean
        get() = synchronized(lock) { buffer.isEmpty() }

    /**
     * True if buffer has reached capacity.
     */
    val isFull: Boolean
        get() = synchronized(lock) { buffer.size >= capacity }

    /**
     * Count of frames currently in buffer that have at least one hand present.
     */
    val activeFramesCount: Int
        get() = synchronized(lock) {
            var count = 0
            for (frame in buffer) {
                if (frame[INDEX_PRES_LEFT] > 0.5f || frame[INDEX_PRES_RIGHT] > 0.5f) {
                    count++
                }
            }
            count
        }

    /**
     * Ingest a single 134-D frame into the rolling window.
     *
     * @param frame 134-D feature vector. A defensive copy is stored.
     * @param hasHands Optional override for hand presence. Defaults to inspecting presence flags at 126 and 127.
     * @return true if hands were present in this frame, false otherwise.
     */
    fun addFrame(
        frame: FloatArray,
        hasHands: Boolean = (frame.size > INDEX_PRES_RIGHT &&
                (frame[INDEX_PRES_LEFT] > 0.5f || frame[INDEX_PRES_RIGHT] > 0.5f))
    ): Boolean = synchronized(lock) {
        require(frame.size == FEATURE_DIM) {
            "Expected frame dimension $FEATURE_DIM, got ${frame.size}"
        }

        if (hasHands) {
            _consecutivePresenceFrames++
            _consecutiveIdleFrames = 0
        } else {
            _consecutiveIdleFrames++
            _consecutivePresenceFrames = 0

            if (autoResetOnIdle && _consecutiveIdleFrames >= idleThreshold) {
                idleResetInternal()
                return false
            }
        }

        if (buffer.size >= capacity) {
            buffer.removeFirst()
        }
        buffer.addLast(frame.clone())
        return hasHands
    }

    /**
     * Returns an array of frames in chronological order (oldest at index 0, newest at index size - 1).
     * Creates defensive copies of each stored frame.
     */
    fun getSnapshot(): Array<FloatArray> = synchronized(lock) {
        val count = buffer.size
        val snapshot = Array(count) { FloatArray(FEATURE_DIM) }
        var i = 0
        for (frame in buffer) {
            System.arraycopy(frame, 0, snapshot[i], 0, FEATURE_DIM)
            i++
        }
        return snapshot
    }

    /**
     * Returns a temporally resampled snapshot of [targetLength] frames (default: 64).
     *
     * - If buffer is empty, returns an array filled with 0.0f.
     * - If buffer has [targetLength] frames, returns snapshot directly (or copies to [destination]).
     * - If buffer has fewer frames (e.g. 16 frames), resamples to [targetLength] via [TemporalResampler.resample].
     *
     * @param targetLength Target sequence length (default 64)
     * @param destination Optional pre-allocated array of shape [targetLength][134] to prevent GC allocations.
     */
    fun getResampledSnapshot(
        targetLength: Int = DEFAULT_CAPACITY,
        destination: Array<FloatArray>? = null
    ): Array<FloatArray> = synchronized(lock) {
        require(targetLength > 0) { "targetLength must be positive: $targetLength" }
        val count = buffer.size
        if (count == 0) {
            val out = destination ?: Array(targetLength) { FloatArray(FEATURE_DIM) }
            for (j in 0 until targetLength) {
                out[j].fill(0.0f)
            }
            return out
        }

        val snapshot = getSnapshot()
        if (count == targetLength) {
            if (destination != null) {
                for (j in 0 until targetLength) {
                    System.arraycopy(snapshot[j], 0, destination[j], 0, FEATURE_DIM)
                }
                return destination
            }
            return snapshot
        }

        return TemporalResampler.resample(snapshot, targetLength, destination)
    }

    /**
     * Returns a defensive copy of the frame at the specified chronological index (0 = oldest).
     */
    fun getFrame(index: Int): FloatArray = synchronized(lock) {
        require(index in 0 until buffer.size) {
            "Index $index out of bounds for buffer size ${buffer.size}"
        }
        var i = 0
        for (frame in buffer) {
            if (i == index) return frame.clone()
            i++
        }
        throw IndexOutOfBoundsException("Index $index out of bounds")
    }

    /**
     * Returns a defensive copy of the most recently added frame, or null if buffer is empty.
     */
    fun getLatestFrame(): FloatArray? = synchronized(lock) {
        return buffer.peekLast()?.clone()
    }

    /**
     * Reset triggered by idle timeout (hand absence).
     * Clears the rolling buffer and resets presence tracking,
     * while retaining the consecutive idle count.
     */
    fun idleReset() = synchronized(lock) {
        idleResetInternal()
    }

    private fun idleResetInternal() {
        buffer.clear()
        _consecutivePresenceFrames = 0
    }

    /**
     * Clears all buffered frames and resets all tracking counters to zero.
     */
    fun clear() = synchronized(lock) {
        buffer.clear()
        _consecutivePresenceFrames = 0
        _consecutiveIdleFrames = 0
    }
}
