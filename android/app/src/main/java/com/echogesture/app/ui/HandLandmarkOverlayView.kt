package com.echogesture.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.echogesture.app.R
import com.echogesture.app.feature.DetectedHandCandidate

/**
 * Custom View overlaid on top of CameraX PreviewView.
 * Renders real-time hand skeleton landmarks and connection lines.
 */
class HandLandmarkOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        // MediaPipe Hand 21-landmark topology connections (pairs of landmark indices)
        val HAND_CONNECTIONS = arrayOf(
            // Thumb
            Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 4),
            // Index finger
            Pair(0, 5), Pair(5, 6), Pair(6, 7), Pair(7, 8),
            // Middle finger
            Pair(0, 9), Pair(9, 10), Pair(10, 11), Pair(11, 12),
            // Ring finger
            Pair(0, 13), Pair(13, 14), Pair(14, 15), Pair(15, 16),
            // Pinky finger
            Pair(0, 17), Pair(17, 18), Pair(18, 19), Pair(19, 20),
            // Palm base / knuckles
            Pair(5, 9), Pair(9, 13), Pair(13, 17)
        )
    }

    private var hands: List<DetectedHandCandidate> = emptyList()
    private var isMirrored: Boolean = false

    private val jointPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.landmark_joint)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val wristPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.landmark_wrist)
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val leftBonePaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.landmark_bone_left)
        strokeWidth = 6f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val rightBonePaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.landmark_bone_right)
        strokeWidth = 6f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        style = Paint.Style.FILL
        isAntiAlias = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    /**
     * Updates detected hands to render. Safe to call from any thread.
     */
    fun setHands(detectedHands: List<DetectedHandCandidate>, isFrontCamera: Boolean = false) {
        this.hands = detectedHands
        this.isMirrored = isFrontCamera
        postInvalidate()
    }

    fun clear() {
        this.hands = emptyList()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (hands.isEmpty()) return

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        for (hand in hands) {
            val coords = hand.landmarks63
            if (coords.size < 63) continue

            val isLeft = hand.label.equals("Left", ignoreCase = true)
            val bonePaint = if (isLeft) leftBonePaint else rightBonePaint

            // 1. Draw connection bones
            for (conn in HAND_CONNECTIONS) {
                val idx1 = conn.first
                val idx2 = conn.second

                val rawX1 = coords[idx1 * 3]
                val rawY1 = coords[idx1 * 3 + 1]
                val rawX2 = coords[idx2 * 3]
                val rawY2 = coords[idx2 * 3 + 1]

                val px1 = if (isMirrored) (1f - rawX1) * viewW else rawX1 * viewW
                val py1 = rawY1 * viewH
                val px2 = if (isMirrored) (1f - rawX2) * viewW else rawX2 * viewW
                val py2 = rawY2 * viewH

                canvas.drawLine(px1, py1, px2, py2, bonePaint)
            }

            // 2. Draw landmark joint dots
            for (i in 0 until 21) {
                val rawX = coords[i * 3]
                val rawY = coords[i * 3 + 1]
                val px = if (isMirrored) (1f - rawX) * viewW else rawX * viewW
                val py = rawY * viewH

                if (i == 0) {
                    canvas.drawCircle(px, py, 10f, wristPaint)
                } else {
                    canvas.drawCircle(px, py, 6f, jointPaint)
                }
            }

            // 3. Draw Hand label tag near wrist
            val wristRawX = coords[0]
            val wristRawY = coords[1]
            val wristPx = if (isMirrored) (1f - wristRawX) * viewW else wristRawX * viewW
            val wristPy = wristRawY * viewH
            val tag = "${hand.label} (${(hand.score * 100).toInt()}%)"
            canvas.drawText(tag, wristPx + 16f, wristPy - 16f, textPaint)
        }
    }
}
