package dev.flutterberlin.flutter_gemma.engines.litertlm

import android.util.Log
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.tool
import dev.flutterberlin.flutter_gemma.engines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import org.json.JSONObject
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

private const val TAG = "LiteRtLmSession"

/** Timeout for Dart-side tool execution (seconds). */
private const val TOOL_EXECUTION_TIMEOUT_SECONDS = 60L

/**
 * Callback interface for executing tools in Dart.
 * Called on the LiteRT-LM inference thread — implementation must be thread-safe.
 */
fun interface ToolExecutor {
    /**
     * Execute a tool and return the result as a JSON string.
     * @param name Tool name
     * @param paramsJson JSON string of tool arguments
     * @return CompletableFuture that resolves with the tool result JSON
     */
    fun execute(name: String, paramsJson: String): CompletableFuture<String>
}

/**
 * LiteRT-LM Session implementation.
 *
 * Key Design Decision: Chunk Buffering
 * - MediaPipe: addQueryChunk() directly on session
 * - LiteRT-LM: sendMessage() takes complete message
 * - Solution: Buffer chunks in StringBuilder, send on generateResponse()
 */
class LiteRtLmSession(
    engine: Engine,
    config: SessionConfig,
    private val resultFlow: MutableSharedFlow<Pair<String, Boolean>>,
    private val errorFlow: MutableSharedFlow<Throwable>,
    private val toolExecutor: ToolExecutor? = null
) : InferenceSession {

    private val conversation: Conversation

    // Extra context for thinking mode (Gemma 4 via Jinja template variable)
    private val extraContext: Map<String, Any> = if (config.enableThinking) {
        mapOf("enable_thinking" to true)
    } else {
        emptyMap()
    }

    // Chunk buffering (MediaPipe compatibility) - thread-safe access
    private val pendingPrompt = StringBuilder()
    private val promptLock = Any()
    @Volatile private var pendingImage: ByteArray? = null
    @Volatile private var pendingAudio: ByteArray? = null

    /** Whether native tools were passed to this session. */
    private val hasNativeTools: Boolean

    init {
        // Build sampler config
        val samplerConfig = SamplerConfig(
            topK = config.topK,
            topP = (config.topP ?: 0.95f).toDouble(),
            temperature = config.temperature.toDouble(),
        )

        // Build native tool providers from JSON definitions
        val toolProviders = config.toolDefinitionsJson?.mapNotNull { jsonStr ->
            try {
                val json = JSONObject(jsonStr)
                val name = json.getString("name")
                val description = json.optString("description", "")
                val parametersJson = json.optJSONObject("parameters")?.toString() ?: "{}"
                object : OpenApiTool {
                    override fun getToolDescriptionJsonString(): String {
                        val toolJson = JSONObject().apply {
                            put("name", name)
                            put("description", description)
                            put("parameters", JSONObject(parametersJson))
                        }
                        return toolJson.toString()
                    }
                    override fun execute(paramsJsonString: String): String {
                        Log.i(TAG, "TOOL_EXECUTE: name=$name, params=$paramsJsonString")

                        // Emit tool-call-started event for UI progress indication
                        val startEvent = JSONObject().apply {
                            put("__native_tool_event__", "started")
                            put("name", name)
                            put("arguments", paramsJsonString)
                        }
                        resultFlow.tryEmit(startEvent.toString() to false)

                        if (toolExecutor != null) {
                            return try {
                                val future = toolExecutor.execute(name, paramsJsonString)
                                val result = future.get(
                                    TOOL_EXECUTION_TIMEOUT_SECONDS, TimeUnit.SECONDS
                                )
                                Log.i(TAG, "TOOL_RESULT: name=$name, result=${result.take(200)}")

                                // Emit tool-call-completed event
                                val endEvent = JSONObject().apply {
                                    put("__native_tool_event__", "completed")
                                    put("name", name)
                                }
                                resultFlow.tryEmit(endEvent.toString() to false)

                                result
                            } catch (e: Exception) {
                                Log.e(TAG, "Tool execution failed: $name", e)

                                // Emit tool-call-error event
                                val errorEvent = JSONObject().apply {
                                    put("__native_tool_event__", "error")
                                    put("name", name)
                                    put("error", e.message ?: e.toString())
                                }
                                resultFlow.tryEmit(errorEvent.toString() to false)

                                """{"error":"Tool execution failed: ${e.message}"}"""
                            }
                        }
                        return """{"error":"no_executor","tool":"$name"}"""
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to parse tool definition: $jsonStr", e)
                null
            }
        } ?: emptyList()

        hasNativeTools = toolProviders.isNotEmpty()

        // Build conversation config — pass tools if available
        val nativeToolProviders = toolProviders.map { tool(it) }

        val conversationConfig = if (hasNativeTools) {
            Log.i(TAG, "Creating conversation with ${nativeToolProviders.size} native tools")
            ConversationConfig(
                samplerConfig = samplerConfig,
                systemInstruction = config.systemInstruction?.let { Contents.of(it) },
                tools = nativeToolProviders,
                automaticToolCalling = true,
            )
        } else {
            ConversationConfig(
                samplerConfig = samplerConfig,
                systemInstruction = config.systemInstruction?.let { Contents.of(it) },
            )
        }

        // Enable constrained decoding when native tools are active.
        // This forces the model to produce valid FC-format tool calls,
        // preventing malformed output. Available in LiteRT-LM 0.10.0+.
        if (hasNativeTools) {
            ExperimentalFlags.enableConversationConstrainedDecoding = true
        }
        conversation = engine.createConversation(conversationConfig)
        // Reset flag after conversation creation (per Gallery pattern)
        ExperimentalFlags.enableConversationConstrainedDecoding = false

        Log.d(TAG, "Created LiteRT-LM conversation with topK=${config.topK}, " +
            "temp=${config.temperature}, nativeTools=$hasNativeTools, " +
            "constrainedDecoding=${hasNativeTools}")
    }

    override fun addQueryChunk(prompt: String) {
        synchronized(promptLock) {
            pendingPrompt.append(prompt)
            Log.v(TAG, "Accumulated chunk: ${prompt.length} chars, total: ${pendingPrompt.length}")
        }
    }

    override fun addImage(imageBytes: ByteArray) {
        synchronized(promptLock) { pendingImage = imageBytes }
        Log.d(TAG, "Added image: ${imageBytes.size} bytes")
    }

    override fun addAudio(audioBytes: ByteArray) {
        synchronized(promptLock) { pendingAudio = audioBytes }
        Log.d(TAG, "Added audio: ${audioBytes.size} bytes")
    }

    override fun generateResponse(): String {
        val message = buildAndConsumeMessage()
        Log.d(TAG, "Generating sync response for message: ${message.toString().length} chars")
        return try {
            val response = if (extraContext.isNotEmpty()) {
                conversation.sendMessage(message, extraContext)
            } else {
                conversation.sendMessage(message)
            }
            val thinking = response.channels["thought"]
            val text = response.toString()
            if (!thinking.isNullOrEmpty()) {
                "<|channel>thought\n$thinking<channel|>$text"
            } else {
                text
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error generating response", e)
            if (!errorFlow.tryEmit(e)) {
                Log.w(TAG, "Error emission dropped (buffer full): ${e.message}")
            }
            throw e
        }
    }

    override fun generateResponseAsync() {
        val message = buildAndConsumeMessage()
        Log.d(TAG, "Generating async response for message: ${message.toString().length} chars")

        try {
            // Create callback with native tool logging support
            val wrappedCallback = object : MessageCallback {
                override fun onMessage(msg: Message) {
                    if (hasNativeTools) {
                        Log.d(TAG, "onMessage: role=${msg.role}, " +
                            "toolCalls=${msg.toolCalls.size}, " +
                            "content=${msg.contents.toString().take(100)}")
                    }
                    // Combine thinking + text into single emission to prevent DROP_OLDEST loss
                    val thinking = msg.channels["thought"]
                    val text = msg.toString()
                    val combined = buildString {
                        if (!thinking.isNullOrEmpty()) {
                            append("<|channel>thought\n$thinking<channel|>")
                        }
                        if (text.isNotEmpty()) {
                            append(text)
                        }
                    }
                    if (combined.isNotEmpty()) {
                        resultFlow.tryEmit(combined to false)
                    }
                }

                override fun onDone() {
                    resultFlow.tryEmit("" to true)
                }

                override fun onError(throwable: Throwable) {
                    Log.e(TAG, "Async generation error", throwable)
                    if (!errorFlow.tryEmit(throwable)) {
                        Log.w(TAG, "Error emission dropped (buffer full): ${throwable.message}")
                    }
                    resultFlow.tryEmit("" to true)
                }
            }

            if (extraContext.isNotEmpty()) {
                conversation.sendMessageAsync(message, wrappedCallback, extraContext)
            } else {
                conversation.sendMessageAsync(message, wrappedCallback)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start async generation", e)
            if (!errorFlow.tryEmit(e)) {
                Log.w(TAG, "Error emission dropped (buffer full): ${e.message}")
            }
            resultFlow.tryEmit("" to true)
        }
    }

    override fun sizeInTokens(prompt: String): Int {
        val estimate = (prompt.length + 3) / 4
        Log.w(TAG, "sizeInTokens: LiteRT-LM does not support token counting. " +
                "Using estimate (~4 chars/token): $estimate tokens for ${prompt.length} chars.")
        return estimate
    }

    override fun cancelGeneration() {
        try {
            conversation.cancelProcess()
            Log.i(TAG, "cancelGeneration: cancelled via Conversation.cancelProcess()")
        } catch (e: Exception) {
            Log.w(TAG, "cancelGeneration: failed to cancel", e)
        }
    }

    override fun close() {
        try {
            conversation.close()
            Log.d(TAG, "Conversation closed")
        } catch (e: Exception) {
            Log.w(TAG, "Error closing conversation", e)
        }
    }

    private fun buildAndConsumeMessage(): Contents {
        val text: String
        val image: ByteArray?
        val audio: ByteArray?
        synchronized(promptLock) {
            text = pendingPrompt.toString()
            pendingPrompt.clear()
            image = pendingImage
            pendingImage = null
            audio = pendingAudio
            pendingAudio = null
        }

        val contents = mutableListOf<Content>()
        image?.let {
            contents.add(Content.ImageBytes(it))
            Log.d(TAG, "Added image: ${it.size} bytes")
        }
        audio?.let {
            contents.add(Content.AudioBytes(it))
            Log.d(TAG, "Added audio: ${it.size} bytes (WAV format)")
        }
        if (text.isNotEmpty() || contents.isEmpty()) {
            contents.add(Content.Text(text))
            Log.d(TAG, "Added text: ${text.length} chars")
        }

        Log.d(TAG, "Building message with ${contents.size} content items")
        return Contents.of(contents)
    }
}
