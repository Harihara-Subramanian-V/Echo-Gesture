package com.echogesture.app.speech

import android.content.Context
import android.media.AudioAttributes
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Abstraction over Android's [TextToSpeech] engine for decoupled unit testing and runtime execution.
 */
interface TtsEngineAdapter {
    fun initialize(context: Context, onInit: (Int) -> Unit)
    fun setLanguage(locale: Locale): Int
    fun isLanguageAvailable(locale: Locale): Int
    fun setPitch(pitch: Float): Int
    fun setSpeechRate(speechRate: Float): Int
    fun setAudioAttributes(audioAttributes: AudioAttributes): Int
    fun setOnUtteranceProgressListener(listener: UtteranceProgressListener?): Int
    fun speak(text: CharSequence, queueMode: Int, params: Bundle?, utteranceId: String): Int
    fun stop(): Int
    fun shutdown()
    fun isSpeaking(): Boolean
}

/**
 * Production implementation backed by Android's native [TextToSpeech] engine.
 * Ensures strictly offline speech synthesis without remote API dependencies.
 */
class DefaultTtsEngineAdapter(
    private val ttsCreator: (Context, TextToSpeech.OnInitListener) -> TextToSpeech = { ctx, listener ->
        TextToSpeech(ctx.applicationContext ?: ctx, listener)
    }
) : TtsEngineAdapter {

    private var tts: TextToSpeech? = null

    override fun initialize(context: Context, onInit: (Int) -> Unit) {
        tts = ttsCreator(context) { status ->
            onInit(status)
        }
    }

    override fun setLanguage(locale: Locale): Int {
        return tts?.setLanguage(locale) ?: TextToSpeech.ERROR
    }

    override fun isLanguageAvailable(locale: Locale): Int {
        return tts?.isLanguageAvailable(locale) ?: TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun setPitch(pitch: Float): Int {
        return tts?.setPitch(pitch) ?: TextToSpeech.ERROR
    }

    override fun setSpeechRate(speechRate: Float): Int {
        return tts?.setSpeechRate(speechRate) ?: TextToSpeech.ERROR
    }

    override fun setAudioAttributes(audioAttributes: AudioAttributes): Int {
        return tts?.setAudioAttributes(audioAttributes) ?: TextToSpeech.ERROR
    }

    override fun setOnUtteranceProgressListener(listener: UtteranceProgressListener?): Int {
        return tts?.setOnUtteranceProgressListener(listener) ?: TextToSpeech.ERROR
    }

    override fun speak(text: CharSequence, queueMode: Int, params: Bundle?, utteranceId: String): Int {
        return tts?.speak(text, queueMode, params, utteranceId) ?: TextToSpeech.ERROR
    }

    override fun stop(): Int {
        return tts?.stop() ?: TextToSpeech.ERROR
    }

    override fun shutdown() {
        tts?.shutdown()
        tts = null
    }

    override fun isSpeaking(): Boolean {
        return tts?.isSpeaking == true
    }
}

/**
 * Thread-safe, offline-capable native Android Text-to-Speech manager for EchoGesture.
 *
 * Implements [ITextToSpeechManager] to provide speech synthesis for recognized gestures.
 * Handles lifecycle initialization, language availability verification with graceful
 * device fallback, speech rate/pitch parameters, accessibility audio routing, and
 * utterance progress callbacks.
 */
class TextToSpeechManager(
    private val context: Context,
    private val engineAdapter: TtsEngineAdapter = DefaultTtsEngineAdapter()
) : ITextToSpeechManager {

    companion object {
        private const val TAG = "TextToSpeechManager"
        const val DEFAULT_PITCH = 1.0f
        const val DEFAULT_SPEECH_RATE = 1.0f
        const val QUEUE_FLUSH = TextToSpeech.QUEUE_FLUSH
        const val QUEUE_ADD = TextToSpeech.QUEUE_ADD
    }

    private val lock = Any()
    private val _isInitialized = AtomicBoolean(false)
    private val _isSpeaking = AtomicBoolean(false)
    private val _isEnabled = AtomicBoolean(true)
    private var isInitializing = false
    private val utteranceCounter = AtomicLong(0L)

    private var lastSpoken: String? = null
    private val pendingInitCallbacks = mutableListOf<(Boolean) -> Unit>()
    private var speechEventListener: SpeechEventListener? = null

    var pitch: Float = DEFAULT_PITCH
        set(value) {
            field = value
            if (_isInitialized.get()) {
                try {
                    engineAdapter.setPitch(value)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to update pitch: ${e.message}")
                }
            }
        }

    var speechRate: Float = DEFAULT_SPEECH_RATE
        set(value) {
            field = value
            if (_isInitialized.get()) {
                try {
                    engineAdapter.setSpeechRate(value)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to update speech rate: ${e.message}")
                }
            }
        }

    override val isInitialized: Boolean
        get() = _isInitialized.get()

    override val isSpeaking: Boolean
        get() = _isSpeaking.get() || (_isInitialized.get() && try {
            engineAdapter.isSpeaking()
        } catch (_: Exception) {
            false
        })

    override var isEnabled: Boolean
        get() = _isEnabled.get()
        set(value) {
            val wasEnabled = _isEnabled.getAndSet(value)
            if (wasEnabled && !value) {
                stop()
            }
        }

    override val lastSpokenText: String?
        get() = synchronized(lock) { lastSpoken }

    override fun setSpeechEventListener(listener: SpeechEventListener?) {
        synchronized(lock) {
            speechEventListener = listener
        }
    }

    private val internalProgressListener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {
            val id = utteranceId.orEmpty()
            _isSpeaking.set(true)
            val listener = synchronized(lock) { speechEventListener }
            try {
                listener?.onSpeechStart(id)
            } catch (e: Exception) {
                Log.w(TAG, "Error in speechEventListener.onSpeechStart: ${e.message}")
            }
        }

        override fun onDone(utteranceId: String?) {
            val id = utteranceId.orEmpty()
            _isSpeaking.set(false)
            val listener = synchronized(lock) { speechEventListener }
            try {
                listener?.onSpeechDone(id)
            } catch (e: Exception) {
                Log.w(TAG, "Error in speechEventListener.onSpeechDone: ${e.message}")
            }
        }

        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String?) {
            val id = utteranceId.orEmpty()
            _isSpeaking.set(false)
            val listener = synchronized(lock) { speechEventListener }
            try {
                listener?.onSpeechError(id, -1)
            } catch (e: Exception) {
                Log.w(TAG, "Error in speechEventListener.onError: ${e.message}")
            }
        }

        override fun onError(utteranceId: String?, errorCode: Int) {
            val id = utteranceId.orEmpty()
            _isSpeaking.set(false)
            val listener = synchronized(lock) { speechEventListener }
            try {
                listener?.onSpeechError(id, errorCode)
            } catch (e: Exception) {
                Log.w(TAG, "Error in speechEventListener.onError: ${e.message}")
            }
        }
    }

    override fun initialize(onReady: ((Boolean) -> Unit)?) {
        synchronized(lock) {
            if (_isInitialized.get()) {
                onReady?.invoke(true)
                return
            }

            if (onReady != null) {
                pendingInitCallbacks.add(onReady)
            }

            if (isInitializing) {
                return
            }
            isInitializing = true

            try {
                engineAdapter.initialize(context) { status ->
                    handleOnInit(status)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception during TTS engine initialization: ${e.message}", e)
                isInitializing = false
                _isInitialized.set(false)
                val callbacks = pendingInitCallbacks.toList()
                pendingInitCallbacks.clear()
                for (cb in callbacks) {
                    try {
                        cb.invoke(false)
                    } catch (ex: Exception) {
                        Log.e(TAG, "Error notifying onReady callback: ${ex.message}", ex)
                    }
                }
            }
        }
    }

    private fun handleOnInit(status: Int) {
        val callbacksToNotify: List<(Boolean) -> Unit>
        val success: Boolean

        synchronized(lock) {
            isInitializing = false
            if (status == TextToSpeech.SUCCESS) {
                success = configureEngine()
                _isInitialized.set(success)
            } else {
                Log.e(TAG, "TTS initialization failed with status: $status")
                _isInitialized.set(false)
                success = false
            }
            callbacksToNotify = pendingInitCallbacks.toList()
            pendingInitCallbacks.clear()
        }

        for (cb in callbacksToNotify) {
            try {
                cb.invoke(success)
            } catch (e: Exception) {
                Log.e(TAG, "Error invoking onReady callback: ${e.message}", e)
            }
        }
    }

    private fun configureEngine(): Boolean {
        return try {
            // 1. Language verification: Locale.US with fallback to device default
            val usStatus = engineAdapter.isLanguageAvailable(Locale.US)
            if (usStatus >= TextToSpeech.LANG_AVAILABLE) {
                val setLangResult = engineAdapter.setLanguage(Locale.US)
                if (setLangResult == TextToSpeech.LANG_MISSING_DATA || setLangResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Locale.US missing data or not supported ($setLangResult). Falling back to default locale.")
                    applyDefaultLocaleFallback()
                } else {
                    Log.i(TAG, "TTS initialized with Locale.US")
                }
            } else {
                Log.w(TAG, "Locale.US unavailable (status: $usStatus). Falling back to default locale.")
                applyDefaultLocaleFallback()
            }

            // 2. Configure speech parameters
            engineAdapter.setPitch(pitch)
            engineAdapter.setSpeechRate(speechRate)

            // 3. Configure audio attributes for accessibility and speech content
            try {
                val audioAttributes = AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
                if (audioAttributes != null) {
                    engineAdapter.setAudioAttributes(audioAttributes)
                }
            } catch (e: Exception) {
                Log.w(TAG, "AudioAttributes setup bypassed: ${e.message}")
            }

            // 4. Register utterance progress listener
            engineAdapter.setOnUtteranceProgressListener(internalProgressListener)

            Log.i(TAG, "EchoGesture TTS manager successfully initialized for offline operation.")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to configure TTS engine: ${e.message}", e)
            false
        }
    }

    private fun applyDefaultLocaleFallback() {
        val defaultLocale = Locale.getDefault()
        val defaultStatus = engineAdapter.isLanguageAvailable(defaultLocale)
        if (defaultStatus >= TextToSpeech.LANG_AVAILABLE) {
            engineAdapter.setLanguage(defaultLocale)
            Log.i(TAG, "TTS successfully configured with default locale: $defaultLocale")
        } else {
            Log.w(TAG, "Default locale $defaultLocale also unavailable (status: $defaultStatus)")
        }
    }

    override fun speak(text: String, queueMode: Int): Boolean {
        if (text.isBlank()) {
            Log.d(TAG, "speak() rejected: text is blank or empty")
            return false
        }

        if (!_isEnabled.get()) {
            Log.d(TAG, "speak() rejected: speech is disabled")
            return false
        }

        if (!_isInitialized.get()) {
            Log.w(TAG, "speak() rejected: TTS manager is not initialized")
            return false
        }

        return synchronized(lock) {
            if (!_isEnabled.get() || !_isInitialized.get()) {
                return false
            }

            try {
                lastSpoken = text
                val utteranceId = "echogesture_utt_${utteranceCounter.incrementAndGet()}"
                val params = Bundle().apply {
                    putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
                }

                val result = engineAdapter.speak(text, queueMode, params, utteranceId)
                if (result == TextToSpeech.SUCCESS) {
                    _isSpeaking.set(true)
                    true
                } else {
                    Log.w(TAG, "TTS speak failed with engine status: $result")
                    false
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception during speak: ${e.message}", e)
                false
            }
        }
    }

    override fun replay(): Boolean {
        val textToReplay = synchronized(lock) { lastSpoken }
        if (textToReplay.isNullOrBlank()) {
            Log.d(TAG, "replay() rejected: no previous utterance recorded")
            return false
        }
        return speak(textToReplay, TextToSpeech.QUEUE_FLUSH)
    }

    override fun stop() {
        synchronized(lock) {
            try {
                if (_isInitialized.get()) {
                    engineAdapter.stop()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Exception during stop: ${e.message}", e)
            } finally {
                _isSpeaking.set(false)
            }
        }
    }

    override fun shutdown() {
        val callbacksToCancel: List<(Boolean) -> Unit>
        synchronized(lock) {
            try {
                stop()
                engineAdapter.setOnUtteranceProgressListener(null)
                engineAdapter.shutdown()
            } catch (e: Exception) {
                Log.w(TAG, "Exception during shutdown: ${e.message}", e)
            } finally {
                _isInitialized.set(false)
                _isSpeaking.set(false)
                isInitializing = false
                callbacksToCancel = pendingInitCallbacks.toList()
                pendingInitCallbacks.clear()
            }
        }

        for (cb in callbacksToCancel) {
            try {
                cb.invoke(false)
            } catch (e: Exception) {
                Log.w(TAG, "Error in canceled initialization callback: ${e.message}", e)
            }
        }
    }

    override fun close() {
        shutdown()
    }
}
