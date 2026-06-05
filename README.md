# ConversationalAI Service

Production-grade conversational AI API built with **FastAPI** and the **OpenAI SDK**.  
Exposes two independent API surfaces: **text chat** and **voice chat**, each supporting both streaming and non-streaming modes.

---

## Architecture

```
project/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── routes/
│   │   ├── chat.py          # POST /chat, POST /chat/stream
│   │   ├── voice.py         # POST /voice/chat, POST /voice/chat/stream
│   │   └── health.py        # GET /health
│   ├── services/
│   │   ├── chat_service.py  # Session memory + LLM orchestration
│   │   ├── voice_service.py # STT → LLM → TTS pipeline
│   │   ├── session_store.py # In-memory conversation memory (swappable)
│   │   ├── stt_service.py   # STT provider interface + Whisper impl
│   │   └── tts_service.py   # TTS provider interface + OpenAI TTS impl
│   ├── clients/
│   │   └── openai_client.py # Async OpenAI wrapper (chat, STT, TTS)
│   ├── models/
│   │   ├── chat.py          # Pydantic request/response models
│   │   ├── voice.py         # Voice metadata models
│   │   └── health.py        # Health check model
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (env-based config)
│   │   ├── exceptions.py    # Domain exceptions
│   │   └── logging.py       # Structured logging setup
│   └── utils/
│       ├── dependencies.py  # FastAPI DI wiring
│       ├── middleware.py     # Logging + exception middleware
│       └── audio.py         # Audio helpers + temp file cleanup
├── requirements.txt
├── .env.example
└── README.md
```

### Key Design Decisions

| Concern | Approach |
|---|---|
| LLM client | `OpenAIClient` wraps `AsyncOpenAI`; all retries/timeouts centralised |
| Session memory | `InMemorySessionStore` (protocol-based, swap with Redis) |
| STT provider | `WhisperSTT` (implements `STTProvider` protocol) |
| TTS provider | `OpenAITTS` (implements `TTSProvider` protocol) |
| DI | FastAPI `Depends()` with `@lru_cache` singletons |
| Streaming text | SSE via `StreamingResponse` + async generator |
| Streaming audio | Chunked `StreamingResponse` via async generator |
| Error handling | Domain exceptions → JSON via `ExceptionHandlerMiddleware` |
| Tracing | `RequestLoggingMiddleware` injects `X-Trace-Id` header |

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo>
cd project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY
```

### 3. Run (development)

```bash
uvicorn app.main:app --reload --port 8000
# Or:
python -m app.main
```

### 4. Run (production)

```bash
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --no-access-log \
  --loop uvloop \
  --http h11
```

### Interactive docs

- Swagger UI: http://localhost:8000/docs  
- ReDoc: http://localhost:8000/redoc  

---

## API Reference

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0","service":"ConversationalAI"}
```

---

### API 1 — Text Chat

#### POST /chat  _(non-streaming)_

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the capital of France?",
    "session_id": "user-abc123"
  }'
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "session_id": "user-abc123",
  "tokens_used": 42
}
```

Optional fields:
```json
{
  "message": "Tell me more.",
  "session_id": "user-abc123",
  "system_prompt": "You are a geography expert. Be very concise.",
  "temperature": 0.3,
  "max_tokens": 512
}
```

---

#### POST /chat/stream  _(Server-Sent Events)_

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "Write a haiku about the ocean.",
    "session_id": "user-abc123"
  }' \
  --no-buffer
```

**SSE stream:**
```
data: {"delta": "Waves", "done": false}
data: {"delta": " crash", "done": false}
data: {"delta": " gently", "done": false}
...
data: {"delta": "", "done": true}
```

**JavaScript client example:**
```javascript
const response = await fetch('/chat/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello!', session_id: 'sess-1' }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (!line.startsWith('data: ')) continue;
    const chunk = JSON.parse(line.slice(6));
    if (chunk.done) break;
    process.stdout.write(chunk.delta);
  }
}
```

---

### API 2 — Voice Chat

#### POST /voice/chat  _(non-streaming)_

```bash
curl -X POST http://localhost:8000/voice/chat \
  -F "audio=@/path/to/recording.wav" \
  -F "session_id=voice-session-1" \
  -F "language=en" \
  -o response.mp3
```

Response headers include:
- `X-Transcript`: STT transcript of your audio
- `X-Response-Text`: LLM text response before TTS
- `X-Session-Id`: echo of your session ID

---

#### POST /voice/chat/stream  _(streaming audio)_

```bash
curl -X POST http://localhost:8000/voice/chat/stream \
  -F "audio=@/path/to/recording.wav" \
  -F "session_id=voice-session-1" \
  -o response_stream.mp3 \
  --no-buffer
```

**Python streaming client:**
```python
import httpx
import asyncio

async def stream_voice():
    async with httpx.AsyncClient(timeout=60) as client:
        with open("recording.wav", "rb") as f:
            async with client.stream(
                "POST",
                "http://localhost:8000/voice/chat/stream",
                files={"audio": ("recording.wav", f, "audio/wav")},
                data={"session_id": "session-1"},
            ) as response:
                with open("reply.mp3", "wb") as out:
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        out.write(chunk)
                        # Feed chunk to audio player here for real-time playback

asyncio.run(stream_voice())
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_TIMEOUT` | `30.0` | Per-request timeout (seconds) |
| `OPENAI_MAX_RETRIES` | `3` | Automatic retry count |
| `STT_MODEL` | `whisper-1` | Whisper model |
| `TTS_MODEL` | `tts-1` | TTS model (`tts-1` or `tts-1-hd`) |
| `TTS_VOICE` | `alloy` | Voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| `TTS_AUDIO_FORMAT` | `mp3` | Output format (`mp3`, `wav`, `opus`, `aac`, `flac`) |
| `SESSION_MAX_MESSAGES` | `20` | Max turns kept in memory per session |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry (seconds) |
| `MAX_AUDIO_SIZE_MB` | `25` | Max uploaded audio file size |
| `DEBUG` | `false` | Enable hot reload and verbose logging |
| `WORKERS` | `4` | Uvicorn worker count (production) |

---

## Swapping Providers

The service is built on protocol interfaces. To swap a provider:

1. Implement the relevant protocol (`STTProvider`, `TTSProvider`)
2. Update `get_voice_service()` in `app/utils/dependencies.py`

No route or service code needs to change.

---

## Production Deployment Notes

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--no-access-log"]
```

### Environment
- Set `DEBUG=false` and `WORKERS` ≥ 2 × CPU cores
- Place behind a reverse proxy (Nginx/Caddy) for TLS termination
- Add `ALLOWED_ORIGINS` to your specific frontend domain(s)
- For session persistence across restarts, replace `InMemorySessionStore` with Redis

### Latency Targets
- Text non-streaming: ~500 ms – 1.5 s (network + LLM TTFT)
- Text streaming TTFT: ~200 – 500 ms
- Voice non-streaming: ~1.5 – 3 s (STT + LLM + TTS serial)
- Voice streaming TTFT: ~1.5 – 2 s (STT blocks; TTS streams)
