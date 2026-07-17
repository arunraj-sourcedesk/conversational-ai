# ConversationalAI Service

Production-grade conversational AI API built with **FastAPI** and the **OpenAI SDK**.  
Exposes the **text chat** API surface supporting non-streaming chat completions.

---

## Architecture

```
project/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── routes/
│   │   ├── chat.py          # POST /chat, GET /chat/sessions, etc.
│   │   └── health.py        # GET /health
│   ├── services/
│   │   ├── chat_service.py  # Session memory + LLM orchestration
│   │   └── session_store.py # In-memory conversation memory (swappable)
│   ├── clients/
│   │   └── openai_client.py # Async OpenAI wrapper (chat completion)
│   ├── models/
│   │   ├── chat.py          # Pydantic request/response models
│   │   └── health.py        # Health check model
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (env-based config)
│   │   ├── exceptions.py    # Domain exceptions
│   │   └── logging.py       # Structured logging setup
│   └── utils/
│       ├── dependencies.py  # FastAPI DI wiring
│       └── middleware.py     # Logging + exception middleware
├── requirements.txt
├── .env.example
└── README.md
```

### Key Design Decisions

| Concern | Approach |
|---|---|
| LLM client | `OpenAIClient` wraps `AsyncOpenAI`; all retries/timeouts centralised |
| Session memory | `InMemorySessionStore` (protocol-based, swap with Redis) |
| DI | FastAPI `Depends()` with `@lru_cache` singletons |
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

### Text Chat

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

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | Your OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `OPENAI_TIMEOUT` | `30.0` | Per-request timeout (seconds) |
| `OPENAI_MAX_RETRIES` | `3` | Automatic retry count |
| `SESSION_MAX_MESSAGES` | `20` | Max turns kept in memory per session |
| `SESSION_TTL_SECONDS` | `3600` | Session expiry (seconds) |
| `DEBUG` | `false` | Enable hot reload and verbose logging |
| `WORKERS` | `4` | Uvicorn worker count (production) |

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
