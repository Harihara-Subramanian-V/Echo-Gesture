package com.echogesture.app.speech

import android.content.Context
import android.content.ContextWrapper
import android.media.AudioAttributes
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Fake test double for [TtsEngineAdapter] allowing comprehensive unit testing on host JVM.
 */
class FakeTtsEngineAdapter : TtsEngineAdapter {
    var initStatusToReturn: Int = TextToSpeech.SUCCESS
    var usLanguageAvailability: Int = TextToSpeech.LANG_AVAILABLE
    var defaultLanguageAvailability: Int = TextToSpeech.LANG_AVAILABLE
    var speakResultToReturn: Int = TextToSpeech.SUCCESS
    var throwOnSpeak: Boolean = false
    var throwOnInitialize: Boolean = false

    var isInitialized = false
    var initializedCallCount = 0
    var lastSetLanguage: Locale? = null
    var lastPitch: Float? = null
    var lastSpeechRate: Float? = null
    var lastAudioAttributes: AudioAttributes? = null
    var progressListener: UtteranceProgressListener? = null
    var isSpeakingFlag = false
    var stopCallCount = 0
    var shutdownCallCount = 0

    val spokenUtterances = mutableListOf<SpokenUtterance>()

    data class SpokenUtterance(
        val text: CharSequence,
        val queueMode: Int,
        val params: Bundle?,
        val utteranceId: String
    )

    override fun initialize(context: Context, onInit: (Int) -> Unit) {
        if (throwOnInitialize) {
            throw RuntimeException("Simulated engine init crash")
        }
        initializedCallCount++
        isInitialized = true
        onInit(initStatusToReturn)
    }

    override fun setLanguage(locale: Locale): Int {
        lastSetLanguage = locale
        return TextToSpeech.LANG_AVAILABLE
    }

    override fun isLanguageAvailable(locale: Locale): Int {
        return if (locale == Locale.US) usLanguageAvailability else defaultLanguageAvailability
    }

    override fun setPitch(pitch: Float): Int {
        lastPitch = pitch
        return TextToSpeech.SUCCESS
    }

    override fun setSpeechRate(speechRate: Float): Int {
        lastSpeechRate = speechRate
        return TextToSpeech.SUCCESS
    }

    override fun setAudioAttributes(audioAttributes: AudioAttributes): Int {
        lastAudioAttributes = audioAttributes
        return TextToSpeech.SUCCESS
    }

    override fun setOnUtteranceProgressListener(listener: UtteranceProgressListener?): Int {
        progressListener = listener
        return TextToSpeech.SUCCESS
    }

    override fun speak(text: CharSequence, queueMode: Int, params: Bundle?, utteranceId: String): Int {
        if (throwOnSpeak) {
            throw RuntimeException("Simulated TTS audio driver failure")
        }
        spokenUtterances.add(SpokenUtterance(text, queueMode, params, utteranceId))
        if (speakResultToReturn == TextToSpeech.SUCCESS) {
            isSpeakingFlag = true
        }
        return speakResultToReturn
    }

    override fun stop(): Int {
        stopCallCount++
        isSpeakingFlag = false
        return TextToSpeech.SUCCESS
    }

    override fun shutdown() {
        shutdownCallCount++
        isSpeakingFlag = false
        isInitialized = false
        progressListener = null
    }

    override fun isSpeaking(): Boolean = isSpeakingFlag
}

/**
 * Host JVM unit test suite for [TextToSpeechManager].
 */
class TextToSpeechManagerTest {

    private lateinit var dummyContext: Context
    private lateinit var fakeEngine: FakeTtsEngineAdapter
    private lateinit var ttsManager: TextToSpeechManager

    @Before
    fun setUp() {
        dummyContext = ContextWrapper(null)
        fakeEngine = FakeTtsEngineAdapter()
        ttsManager = TextToSpeechManager(
            context = dummyContext,
            engineAdapter = fakeEngine
        )
    }

    // -------------------------------------------------------------------------
    // 1. Initial State
    // -------------------------------------------------------------------------

    @Test
    fun testInitialState() {
        assertFalse(ttsManager.isInitialized)
        assertFalse(ttsManager.isSpeaking)
        assertTrue(ttsManager.isEnabled)
        assertNull(ttsManager.lastSpokenText)
    }

    // -------------------------------------------------------------------------
    // 2. Initialization Flow
    // -------------------------------------------------------------------------

    @Test
    fun testSuccessfulInitializationWithUsLocale() {
        var readyResult: Boolean? = null
        ttsManager.initialize { success ->
            readyResult = success
        }

        assertTrue(readyResult == true)
        assertTrue(ttsManager.isInitialized)
        assertEquals(Locale.US, fakeEngine.lastSetLanguage)
        assertEquals(1.0f, fakeEngine.lastPitch)
        assertEquals(1.0f, fakeEngine.lastSpeechRate)
        assertNotNull(fakeEngine.progressListener)
    }

    @Test
    fun testInitializationFallbackToDefaultLocaleWhenUsNotAvailable() {
        val originalLocale = Locale.getDefault()
        try {
            Locale.setDefault(Locale.UK)
            fakeEngine.usLanguageAvailability = TextToSpeech.LANG_NOT_SUPPORTED
            fakeEngine.defaultLanguageAvailability = TextToSpeech.LANG_AVAILABLE

            var readyResult: Boolean? = null
            ttsManager.initialize { success ->
                readyResult = success
            }

            assertTrue(readyResult == true)
            assertTrue(ttsManager.isInitialized)
            assertEquals(Locale.UK, fakeEngine.lastSetLanguage)
        } finally {
            Locale.setDefault(originalLocale)
        }
    }

    @Test
    fun testInitializationWhenNoLanguageAvailableGracefullyCompletes() {
        fakeEngine.usLanguageAvailability = TextToSpeech.LANG_NOT_SUPPORTED
        fakeEngine.defaultLanguageAvailability = TextToSpeech.LANG_NOT_SUPPORTED

        var readyResult: Boolean? = null
        ttsManager.initialize { success ->
            readyResult = success
        }

        // Configuration completes without crash
        assertTrue(readyResult == true)
        assertTrue(ttsManager.isInitialized)
    }

    @Test
    fun testInitializationFailureWhenEngineReportsError() {
        fakeEngine.initStatusToReturn = TextToSpeech.ERROR

        var readyResult: Boolean? = null
        ttsManager.initialize { success ->
            readyResult = success
        }

        assertFalse(readyResult == true)
        assertFalse(ttsManager.isInitialized)
        assertFalse(ttsManager.speak("Hello"))
    }

    @Test
    fun testRedundantInitializationCallTriggersCallbackImmediately() {
        ttsManager.initialize()
        assertTrue(ttsManager.isInitialized)
        assertEquals(1, fakeEngine.initializedCallCount)

        var secondReadyResult: Boolean? = null
        ttsManager.initialize { success ->
            secondReadyResult = success
        }

        assertTrue(secondReadyResult == true)
        // Should not have re-initialized the underlying engine
        assertEquals(1, fakeEngine.initializedCallCount)
    }

    @Test
    fun testMultipleQueuedInitializationCallbacks() {
        var call1 = false
        var call2 = false

        ttsManager.initialize { call1 = it }
        ttsManager.initialize { call2 = it }

        assertTrue(call1)
        assertTrue(call2)
        assertTrue(ttsManager.isInitialized)
    }

    // -------------------------------------------------------------------------
    // 3. Validation & Speech Control
    // -------------------------------------------------------------------------

    @Test
    fun testSpeakRejectsBlankOrEmptyStrings() {
        ttsManager.initialize()

        assertFalse(ttsManager.speak(""))
        assertFalse(ttsManager.speak("   "))
        assertFalse(ttsManager.speak("\n\t"))
        assertTrue(fakeEngine.spokenUtterances.isEmpty())
        assertNull(ttsManager.lastSpokenText)
    }

    @Test
    fun testSpeakRejectsWhenNotInitialized() {
        assertFalse(ttsManager.isInitialized)
        assertFalse(ttsManager.speak("Thank you"))
        assertTrue(fakeEngine.spokenUtterances.isEmpty())
    }

    @Test
    fun testSpeakRejectsWhenDisabled() {
        ttsManager.initialize()
        ttsManager.isEnabled = false

        assertFalse(ttsManager.speak("Thank you"))
        assertTrue(fakeEngine.spokenUtterances.isEmpty())
    }

    @Test
    fun testSuccessfulSpeakAndCaching() {
        ttsManager.initialize()

        val success = ttsManager.speak("Hello EchoGesture", queueMode = TextToSpeech.QUEUE_FLUSH)
        assertTrue(success)
        assertEquals("Hello EchoGesture", ttsManager.lastSpokenText)
        assertEquals(1, fakeEngine.spokenUtterances.size)

        val utterance = fakeEngine.spokenUtterances.first()
        assertEquals("Hello EchoGesture", utterance.text)
        assertEquals(TextToSpeech.QUEUE_FLUSH, utterance.queueMode)
        assertTrue(utterance.utteranceId.startsWith("echogesture_utt_"))
    }

    // -------------------------------------------------------------------------
    // 4. Replay Capabilities
    // -------------------------------------------------------------------------

    @Test
    fun testReplayWithoutPriorSpeechReturnsFalse() {
        ttsManager.initialize()
        assertNull(ttsManager.lastSpokenText)

        assertFalse(ttsManager.replay())
        assertTrue(fakeEngine.spokenUtterances.isEmpty())
    }

    @Test
    fun testReplaySpeaksLastSpokenText() {
        ttsManager.initialize()

        ttsManager.speak("First Sign")
        assertEquals(1, fakeEngine.spokenUtterances.size)

        val replaySuccess = ttsManager.replay()
        assertTrue(replaySuccess)
        assertEquals(2, fakeEngine.spokenUtterances.size)
        assertEquals("First Sign", fakeEngine.spokenUtterances[1].text)
        assertEquals(TextToSpeech.QUEUE_FLUSH, fakeEngine.spokenUtterances[1].queueMode)
    }

    @Test
    fun testReplayFailsWhenDisabled() {
        ttsManager.initialize()
        ttsManager.speak("First Sign")

        ttsManager.isEnabled = false
        assertFalse(ttsManager.replay())
        assertEquals(1, fakeEngine.spokenUtterances.size)
    }

    // -------------------------------------------------------------------------
    // 5. Utterance Progress & Speaking State
    // -------------------------------------------------------------------------

    @Test
    fun testUtteranceProgressListenerTracksSpeakingState() {
        ttsManager.initialize()
        val listener = fakeEngine.progressListener
        assertNotNull(listener)

        var startedId = ""
        var doneId = ""
        var errorId = ""
        var errorCodeReported = 0

        ttsManager.setSpeechEventListener(object : SpeechEventListener {
            override fun onSpeechStart(utteranceId: String) {
                startedId = utteranceId
            }

            override fun onSpeechDone(utteranceId: String) {
                doneId = utteranceId
            }

            override fun onSpeechError(utteranceId: String, errorCode: Int) {
                errorId = utteranceId
                errorCodeReported = errorCode
            }
        })

        // On Start
        listener!!.onStart("utt_123")
        assertTrue(ttsManager.isSpeaking)
        assertEquals("utt_123", startedId)

        // On Done
        listener.onDone("utt_123")
        assertFalse(ttsManager.isSpeaking)
        assertEquals("utt_123", doneId)

        // On Error
        listener.onStart("utt_456")
        assertTrue(ttsManager.isSpeaking)
        listener.onError("utt_456", -2)
        assertFalse(ttsManager.isSpeaking)
        assertEquals("utt_456", errorId)
        assertEquals(-2, errorCodeReported)
    }

    // -------------------------------------------------------------------------
    // 6. Stop, Disable, and Shutdown
    // -------------------------------------------------------------------------

    @Test
    fun testStopHaltsEngineAndResetsSpeakingState() {
        ttsManager.initialize()
        ttsManager.speak("Gesture speaking")
        assertTrue(ttsManager.isSpeaking)

        ttsManager.stop()
        assertEquals(1, fakeEngine.stopCallCount)
        assertFalse(ttsManager.isSpeaking)
    }

    @Test
    fun testDisablingTtsHaltsActiveSpeaking() {
        ttsManager.initialize()
        ttsManager.speak("Gesture speaking")
        assertTrue(ttsManager.isSpeaking)

        ttsManager.isEnabled = false
        assertEquals(1, fakeEngine.stopCallCount)
        assertFalse(ttsManager.isSpeaking)
    }

    @Test
    fun testShutdownResetsStateAndReleasesEngine() {
        ttsManager.initialize()
        ttsManager.speak("Active gesture")

        ttsManager.shutdown()
        assertFalse(ttsManager.isInitialized)
        assertFalse(ttsManager.isSpeaking)
        assertEquals(1, fakeEngine.shutdownCallCount)
        assertNull(fakeEngine.progressListener)

        // Further speak attempts fail
        assertFalse(ttsManager.speak("Post shutdown"))
    }

    @Test
    fun testAutoCloseableCloseDelegatesToShutdown() {
        ttsManager.initialize()
        ttsManager.close()

        assertFalse(ttsManager.isInitialized)
        assertEquals(1, fakeEngine.shutdownCallCount)
    }

    // -------------------------------------------------------------------------
    // 7. Robust Error Handling & Thread Safety
    // -------------------------------------------------------------------------

    @Test
    fun testExceptionDuringSpeakHandledSafely() {
        ttsManager.initialize()
        fakeEngine.throwOnSpeak = true

        val result = ttsManager.speak("Test Exception")
        assertFalse(result)
        // Should not crash the caller
    }

    @Test
    fun testExceptionDuringInitHandledSafely() {
        fakeEngine.throwOnInitialize = true
        var callbackCalled = false
        var callbackSuccess = true

        ttsManager.initialize { success ->
            callbackCalled = true
            callbackSuccess = success
        }

        assertTrue(callbackCalled)
        assertFalse(callbackSuccess)
        assertFalse(ttsManager.isInitialized)
    }

    @Test
    fun testConcurrentAccessThreadSafety() {
        ttsManager.initialize()
        val numThreads = 8
        val iterationsPerThread = 50
        val latch = CountDownLatch(numThreads)
        val successCount = AtomicInteger(0)

        for (t in 0 until numThreads) {
            Thread {
                try {
                    for (i in 0 until iterationsPerThread) {
                        if (i % 5 == 0) {
                            ttsManager.stop()
                        } else if (i % 7 == 0) {
                            ttsManager.replay()
                        } else {
                            if (ttsManager.speak("Phrase $t-$i")) {
                                successCount.incrementAndGet()
                            }
                        }
                    }
                } finally {
                    latch.countDown()
                }
            }.start()
        }

        val completed = latch.await(5, TimeUnit.SECONDS)
        assertTrue("Concurrent operations should complete within timeout", completed)
        assertTrue(successCount.get() > 0)
    }
}
