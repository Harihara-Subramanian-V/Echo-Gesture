package com.echogesture.app.feature

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import kotlin.math.sqrt

/**
 * Unit test suite for [FeatureExtractor134D].
 *
 * Validates the exact 134-D feature vector assembly and invariants specified in
 * Phase 7B.2 and the Stage 1 Android Parity test contract:
 * 1. Left hand only: right slot zeroed, inter-hand features strictly zero.
 * 2. Right hand only: left slot zeroed, inter-hand features strictly zero.
 * 3. Both hands present: exact math for d_geom, dx_geom, dy_geom under 640x480.
 * 4. Neither hand present: entire 134-D vector is all zeros.
 * 5. Aspect ratio test: aspect-ratio scaling invariance (1280x720 vs 1920x1080 produce identical geometry).
 * 6. Zero-masking on dirty inputs and buffer reuse.
 * 7. In-place extraction to custom destination buffers.
 */
class FeatureExtractor134DTest {

    private lateinit var extractor: FeatureExtractor134D

    @Before
    fun setUp() {
        extractor = FeatureExtractor134D(preallocateOutput = false)
    }

    private fun createCanonicalLandmarks(wristX: Float, wristY: Float, wristZ: Float = 0.0f): FloatArray {
        val lms = FloatArray(FeatureExtractor134D.COORDS_PER_HAND)
        // Canonical anatomical offset pattern matching Python generate_hand_landmarks
        val offsets = floatArrayOf(
            0.00f,  0.00f,  0.00f,  // 0: Wrist
            -0.03f, -0.02f, -0.01f, // 1: Thumb CMC
            -0.05f, -0.05f, -0.02f, // 2: Thumb MCP
            -0.07f, -0.07f, -0.02f, // 3: Thumb IP
            -0.08f, -0.09f, -0.03f, // 4: Thumb Tip
            -0.03f, -0.08f, -0.01f, // 5: Index MCP
            -0.04f, -0.12f, -0.02f, // 6: Index PIP
            -0.04f, -0.15f, -0.02f, // 7: Index DIP
            -0.04f, -0.18f, -0.03f, // 8: Index Tip
            -0.01f, -0.09f,  0.00f, // 9: Middle MCP
            -0.01f, -0.13f, -0.01f, // 10: Middle PIP
            -0.01f, -0.16f, -0.02f, // 11: Middle DIP
            -0.01f, -0.19f, -0.02f, // 12: Middle Tip
             0.01f, -0.08f, -0.01f, // 13: Ring MCP
             0.02f, -0.12f, -0.02f, // 14: Ring PIP
             0.02f, -0.15f, -0.02f, // 15: Ring DIP
             0.02f, -0.18f, -0.02f, // 16: Ring Tip
             0.03f, -0.07f, -0.01f, // 17: Pinky MCP
             0.05f, -0.10f, -0.02f, // 18: Pinky PIP
             0.06f, -0.13f, -0.02f, // 19: Pinky DIP
             0.07f, -0.15f, -0.03f  // 20: Pinky Tip
        )
        for (i in 0 until FeatureExtractor134D.LANDMARKS_PER_HAND) {
            lms[i * 3 + 0] = wristX + offsets[i * 3 + 0]
            lms[i * 3 + 1] = wristY + offsets[i * 3 + 1]
            lms[i * 3 + 2] = wristZ + offsets[i * 3 + 2]
        }
        return lms
    }

    // =========================================================================
    // Requirement 1: Left hand only
    // =========================================================================
    @Test
    fun testLeftHandOnly_RightSlotZeroedAndInterHandFeaturesZeroed() {
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val rightLms = FloatArray(FeatureExtractor134D.COORDS_PER_HAND)

        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 1.0f,
            presRight = 0.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.0f
        )

        val feat = extractor.extract(slots, imageWidth = 640, imageHeight = 480)

        assertEquals(FeatureExtractor134D.FEATURE_DIM, feat.size)

        // Slot 0 (0..62) populated with left hand landmarks
        assertArrayEquals(leftLms, feat.copyOfRange(0, 63), 1e-6f)

        // Slot 1 (63..125) strictly zero-masked
        assertArrayEquals(FloatArray(63), feat.copyOfRange(63, 126), 1e-6f)

        // Presence & scores
        assertEquals(1.0f, feat[FeatureExtractor134D.INDEX_PRES_LEFT], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_PRES_RIGHT], 1e-6f)
        assertEquals(0.95f, feat[FeatureExtractor134D.INDEX_SCORE_LEFT], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_SCORE_RIGHT], 1e-6f)

        // Inter-hand co-presence & geometry strictly zeroed
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_BOTH_PRESENT], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_DIST], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 1e-6f)
    }

    // =========================================================================
    // Requirement 2: Right hand only
    // =========================================================================
    @Test
    fun testRightHandOnly_LeftSlotZeroedAndInterHandFeaturesZeroed() {
        val leftLms = FloatArray(FeatureExtractor134D.COORDS_PER_HAND)
        val rightLms = createCanonicalLandmarks(wristX = 0.70f, wristY = 0.60f)

        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 0.0f,
            presRight = 1.0f,
            scoreLeft = 0.0f,
            scoreRight = 0.92f
        )

        val feat = extractor.extract(slots, imageWidth = 640, imageHeight = 480)

        assertEquals(FeatureExtractor134D.FEATURE_DIM, feat.size)

        // Slot 0 (0..62) strictly zero-masked
        assertArrayEquals(FloatArray(63), feat.copyOfRange(0, 63), 1e-6f)

        // Slot 1 (63..125) populated with right hand landmarks
        assertArrayEquals(rightLms, feat.copyOfRange(63, 126), 1e-6f)

        // Presence & scores
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_PRES_LEFT], 1e-6f)
        assertEquals(1.0f, feat[FeatureExtractor134D.INDEX_PRES_RIGHT], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_SCORE_LEFT], 1e-6f)
        assertEquals(0.92f, feat[FeatureExtractor134D.INDEX_SCORE_RIGHT], 1e-6f)

        // Inter-hand co-presence & geometry strictly zeroed
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_BOTH_PRESENT], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_DIST], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 1e-6f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 1e-6f)
    }

    // =========================================================================
    // Requirement 3: Both hands present under 640x480
    // =========================================================================
    @Test
    fun testBothHandsPresent_VerifyExactGeometryUnder640x480() {
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val rightLms = createCanonicalLandmarks(wristX = 0.70f, wristY = 0.50f)

        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 1.0f,
            presRight = 1.0f,
            scoreLeft = 0.98f,
            scoreRight = 0.94f
        )

        val feat = extractor.extract(slots, imageWidth = 640, imageHeight = 480)

        // Slot landmarks
        assertArrayEquals(leftLms, feat.copyOfRange(0, 63), 1e-6f)
        assertArrayEquals(rightLms, feat.copyOfRange(63, 126), 1e-6f)

        // Presence & scores
        assertEquals(1.0f, feat[FeatureExtractor134D.INDEX_PRES_LEFT], 1e-6f)
        assertEquals(1.0f, feat[FeatureExtractor134D.INDEX_PRES_RIGHT], 1e-6f)
        assertEquals(0.98f, feat[FeatureExtractor134D.INDEX_SCORE_LEFT], 1e-6f)
        assertEquals(0.94f, feat[FeatureExtractor134D.INDEX_SCORE_RIGHT], 1e-6f)
        assertEquals(1.0f, feat[FeatureExtractor134D.INDEX_BOTH_PRESENT], 1e-6f)

        // Geometry verification:
        // dx_norm = 0.70 - 0.30 = 0.40
        // dy_norm = 0.50 - 0.60 = -0.10
        // dx_px = 0.40 * 640 = 256.0
        // dy_px = -0.10 * 480 = -48.0
        // dist_px = sqrt(256^2 + (-48)^2) = sqrt(65536 + 2304) = sqrt(67840) ≈ 260.46113
        // diag = sqrt(640^2 + 480^2) = 800.0
        // expected d_geom = 260.46113 / 800.0 ≈ 0.3255764
        // expected dx_geom = 256.0 / 800.0 = 0.32
        // expected dy_geom = -48.0 / 800.0 = -0.06
        val expectedDist = (sqrt(256.0 * 256.0 + (-48.0) * (-48.0)) / 800.0).toFloat()
        val expectedDx = (256.0 / 800.0).toFloat()
        val expectedDy = (-48.0 / 800.0).toFloat()

        assertEquals(expectedDist, feat[FeatureExtractor134D.INDEX_INTER_DIST], 1e-5f)
        assertEquals(expectedDx, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 1e-5f)
        assertEquals(expectedDy, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 1e-5f)
    }

    // =========================================================================
    // Requirement 4: Neither hand present
    // =========================================================================
    @Test
    fun testNeitherHandPresent_EntireVectorIsAllZeros() {
        val emptySlots = ResolvedHandSlots(
            slot0Landmarks = FloatArray(FeatureExtractor134D.COORDS_PER_HAND),
            slot1Landmarks = FloatArray(FeatureExtractor134D.COORDS_PER_HAND),
            presLeft = 0.0f,
            presRight = 0.0f,
            scoreLeft = 0.0f,
            scoreRight = 0.0f
        )

        val feat = extractor.extract(emptySlots, imageWidth = 640, imageHeight = 480)

        assertEquals(FeatureExtractor134D.FEATURE_DIM, feat.size)
        assertArrayEquals(FloatArray(FeatureExtractor134D.FEATURE_DIM), feat, 0.0f)
    }

    // =========================================================================
    // Requirement 5: Aspect ratio test (1280x720 vs 1920x1080 invariance)
    // =========================================================================
    @Test
    fun testAspectRatioInvariance_1280x720Vs1920x1080ProduceIdenticalGeometry() {
        // Same normalized coordinates: Left wrist (0.30, 0.50), Right wrist (0.70, 0.50)
        // dx_norm = 0.40, dy_norm = 0.00
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.50f)
        val rightLms = createCanonicalLandmarks(wristX = 0.70f, wristY = 0.50f)

        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 1.0f,
            presRight = 1.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.95f
        )

        val feat640 = extractor.extract(slots, imageWidth = 640, imageHeight = 480)
        val feat1280 = extractor.extract(slots, imageWidth = 1280, imageHeight = 720)
        val feat1920 = extractor.extract(slots, imageWidth = 1920, imageHeight = 1080)

        // 1. 640x480 (4:3 aspect ratio, diag = 800):
        // dx_px = 0.4 * 640 = 256, dy_px = 0, diag = 800 => 256 / 800 = 0.3200
        assertEquals(0.3200f, feat640[FeatureExtractor134D.INDEX_INTER_DIST], 1e-4f)
        assertEquals(0.3200f, feat640[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 1e-4f)
        assertEquals(0.0f, feat640[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 1e-5f)

        // 2. 1280x720 (16:9 aspect ratio, diag = sqrt(1280^2 + 720^2) ≈ 1468.60478):
        // dx_px = 0.4 * 1280 = 512, diag ≈ 1468.60478 => 512 / 1468.60478 ≈ 0.3486302
        val expected1280Dist = (512.0 / sqrt(1280.0 * 1280.0 + 720.0 * 720.0)).toFloat()
        assertEquals(expected1280Dist, feat1280[FeatureExtractor134D.INDEX_INTER_DIST], 1e-5f)
        assertEquals(expected1280Dist, feat1280[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 1e-5f)
        assertEquals(0.0f, feat1280[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 1e-5f)

        // 3. 1920x1080 (16:9 aspect ratio, diag = sqrt(1920^2 + 1080^2) ≈ 2202.90717):
        // dx_px = 0.4 * 1920 = 768, diag ≈ 2202.90717 => 768 / 2202.90717 ≈ 0.3486302
        val expected1920Dist = (768.0 / sqrt(1920.0 * 1920.0 + 1080.0 * 1080.0)).toFloat()
        assertEquals(expected1920Dist, feat1920[FeatureExtractor134D.INDEX_INTER_DIST], 1e-5f)

        // 4. Invariance Assertion: 1280x720 and 1920x1080 share the exact 16:9 aspect ratio,
        // so their diagonal-normalized inter-wrist distance and offsets MUST be identical!
        assertEquals(
            feat1280[FeatureExtractor134D.INDEX_INTER_DIST],
            feat1920[FeatureExtractor134D.INDEX_INTER_DIST],
            1e-5f
        )
        assertEquals(
            feat1280[FeatureExtractor134D.INDEX_INTER_OFFSET_DX],
            feat1920[FeatureExtractor134D.INDEX_INTER_OFFSET_DX],
            1e-5f
        )
        assertEquals(
            feat1280[FeatureExtractor134D.INDEX_INTER_OFFSET_DY],
            feat1920[FeatureExtractor134D.INDEX_INTER_OFFSET_DY],
            1e-5f
        )

        // 5. Cross-aspect ratio distinction: 4:3 (0.3200) vs 16:9 (0.3486) must differ
        assertNotEquals(
            feat640[FeatureExtractor134D.INDEX_INTER_DIST],
            feat1280[FeatureExtractor134D.INDEX_INTER_DIST],
            1e-3f
        )
    }

    // =========================================================================
    // Requirement 6: Zero-masking on dirty inputs
    // =========================================================================
    @Test
    fun testZeroMaskingOnDirtyInputs() {
        val dirtyLandmarks = FloatArray(FeatureExtractor134D.COORDS_PER_HAND) { 0.77f }
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)

        // Slot 1 has non-zero landmarks and score, but presRight == 0.0f
        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = dirtyLandmarks,
            presLeft = 1.0f,
            presRight = 0.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.88f // Dirty score when absent
        )

        val feat = extractor.extract(slots, imageWidth = 640, imageHeight = 480)

        // Slot 1 must be strictly zero-masked despite dirtyLandmarks
        for (i in FeatureExtractor134D.SLOT1_START until FeatureExtractor134D.SLOT1_END) {
            assertEquals("Index $i should be 0.0f", 0.0f, feat[i], 0.0f)
        }
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_PRES_RIGHT], 0.0f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_SCORE_RIGHT], 0.0f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_BOTH_PRESENT], 0.0f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_DIST], 0.0f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 0.0f)
        assertEquals(0.0f, feat[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 0.0f)
    }

    // =========================================================================
    // Requirement 7: Buffer reuse without stale state leakage
    // =========================================================================
    @Test
    fun testBufferReuseNoLeakage() {
        val reusableExtractor = FeatureExtractor134D(preallocateOutput = true)

        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val rightLms = createCanonicalLandmarks(wristX = 0.70f, wristY = 0.50f)

        // Frame 1: Both hands present
        val frame1Slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 1.0f,
            presRight = 1.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.95f
        )
        val feat1 = reusableExtractor.extract(frame1Slots, imageWidth = 640, imageHeight = 480)
        assertEquals(1.0f, feat1[FeatureExtractor134D.INDEX_BOTH_PRESENT], 0.0f)
        assertTrue(feat1[FeatureExtractor134D.INDEX_INTER_DIST] > 0.0f)

        // Frame 2: Left hand only (Right hand disappeared)
        val frame2Slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = FloatArray(63),
            presLeft = 1.0f,
            presRight = 0.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.0f
        )
        val feat2 = reusableExtractor.extract(frame2Slots, imageWidth = 640, imageHeight = 480)

        // Crucial: verify that Slot 1 and geometric features are completely zeroed out
        for (i in FeatureExtractor134D.SLOT1_START until FeatureExtractor134D.SLOT1_END) {
            assertEquals("Index $i must be zeroed in frame 2", 0.0f, feat2[i], 0.0f)
        }
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_PRES_RIGHT], 0.0f)
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_SCORE_RIGHT], 0.0f)
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_BOTH_PRESENT], 0.0f)
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_INTER_DIST], 0.0f)
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_INTER_OFFSET_DX], 0.0f)
        assertEquals(0.0f, feat2[FeatureExtractor134D.INDEX_INTER_OFFSET_DY], 0.0f)
    }

    // =========================================================================
    // Requirement 8: Custom destination buffer extraction
    // =========================================================================
    @Test
    fun testExtractToCustomDestination() {
        val dest = FloatArray(FeatureExtractor134D.FEATURE_DIM) { -999.0f }

        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = FloatArray(63),
            presLeft = 1.0f,
            presRight = 0.0f,
            scoreLeft = 0.90f,
            scoreRight = 0.0f
        )

        val result = extractor.extractTo(slots, imageWidth = 640, imageHeight = 480, destination = dest)

        assertTrue(result === dest)
        // Verify no -999.0f sentinel remains
        for (i in 0 until FeatureExtractor134D.FEATURE_DIM) {
            assertTrue("Index $i was not written to: ${dest[i]}", dest[i] != -999.0f)
        }
    }

    // =========================================================================
    // Requirement 9: Invalid dimensions handling
    // =========================================================================
    @Test
    fun testZeroOrNegativeDimensions_ZeroesGeometryFeatures() {
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val rightLms = createCanonicalLandmarks(wristX = 0.70f, wristY = 0.50f)

        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = rightLms,
            presLeft = 1.0f,
            presRight = 1.0f,
            scoreLeft = 0.95f,
            scoreRight = 0.95f
        )

        val featZeroWidth = extractor.extract(slots, imageWidth = 0, imageHeight = 480)
        assertEquals(0.0f, featZeroWidth[FeatureExtractor134D.INDEX_BOTH_PRESENT], 0.0f)
        assertEquals(0.0f, featZeroWidth[FeatureExtractor134D.INDEX_INTER_DIST], 0.0f)

        val featNegHeight = extractor.extract(slots, imageWidth = 640, imageHeight = -1)
        assertEquals(0.0f, featNegHeight[FeatureExtractor134D.INDEX_BOTH_PRESENT], 0.0f)
        assertEquals(0.0f, featNegHeight[FeatureExtractor134D.INDEX_INTER_DIST], 0.0f)
    }

    // =========================================================================
    // Requirement 10: Backward compatible alias
    // =========================================================================
    @Test
    fun testAssembleAliasProducesIdenticalVector() {
        val leftLms = createCanonicalLandmarks(wristX = 0.30f, wristY = 0.60f)
        val slots = ResolvedHandSlots(
            slot0Landmarks = leftLms,
            slot1Landmarks = FloatArray(63),
            presLeft = 1.0f,
            presRight = 0.0f,
            scoreLeft = 0.90f,
            scoreRight = 0.0f
        )

        val extracted = extractor.extract(slots, 640, 480)
        val assembled = extractor.assemble(slots, 640, 480)

        assertArrayEquals(extracted, assembled, 0.0f)
    }
}
