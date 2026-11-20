package com.echogesture.app.speech

/**
 * Callbacks for utterance synthesis lifecycle events.
 */
interface SpeechEventListener {
    fun onSpeechStart(utteranceId: String)
    fun onSpeechDone(utteranceId: String)
    fun onSpeechError(utteranceId: String, errorCode: Int)
}

/**
 * Text-to-Speech manager interface contract for EchoGesture Phase 7D.
 */
interface ITextToSpeechManager : AutoCloseable {
    companion object {
        const val QUEUE_FLUSH = 0
        const val QUEUE_ADD = 1
    }

    val isInitialized: Boolean
    val isSpeaking: Boolean
    var isEnabled: Boolean
    val lastSpokenText: String?
    fun initialize(onReady: ((Boolean) -> Unit)? = null)
    fun speak(text: String, queueMode: Int = QUEUE_FLUSH): Boolean
    fun replay(): Boolean
    fun stop()
    fun shutdown()
    fun setSpeechEventListener(listener: SpeechEventListener?) {}

    override fun close() {
        shutdown()
    }
}
