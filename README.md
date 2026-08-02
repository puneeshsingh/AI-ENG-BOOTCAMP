# Ask API

A small FastAPI service that streams answers from OpenAI's `gpt-4o-mini`, with a Streamlit
front end, request guardrails, and per-call cost tracking.

## Setup

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Running

Start the API:

```powershell
uvicorn main:app --reload
```

Start the UI (in a second terminal):

```powershell
streamlit run ui/app.py
```

The UI expects the API at `http://127.0.0.1:8000`.

## API

### `GET /health`

Liveness check. Returns `{"status": "ok"}`.

### `POST /ask`

```json
{ "question": "What is the capital of France?" }
```

Returns the answer plus latency, token, and cost metrics:

```json
{
  "answer": "...",
  "model": "gpt-4o-mini",
  "metrics": {
    "time_to_first_token_sec": 0.412,
    "total_latency_sec": 1.87,
    "tokens_per_sec": 34.2,
    "prompt_tokens": 12,
    "completion_tokens": 64,
    "total_tokens": 76,
    "cost_usd": 0.000040
  }
}
```

## Guardrails

Four guardrails run on every `/ask` call, in this order:

| Guardrail | Behavior | Trigger response |
|---|---|---|
| **Input validation** | Rejects blank questions and caps length at `MAX_QUESTION_CHARS` (4000 chars) | `422 Unprocessable Entity` |
| **Rate limiting** | In-memory sliding window, `RATE_LIMIT_MAX_REQUESTS` (5) requests per `RATE_LIMIT_WINDOW_SECONDS` (60s) per client IP | `429 Too Many Requests` |
| **Content moderation** | Every question is checked against OpenAI's moderation endpoint before it reaches the model | `400 Bad Request` with the flagged category names |
| **Cost/token cap** | Completions are capped at `MAX_COMPLETION_TOKENS` (500) via `max_tokens`, bounding worst-case spend per call alongside the input length cap | Truncated response, no error |

All four are covered by automated tests in `tests/test_guardrails.py` (OpenAI calls are mocked,
so the suite runs offline):

```powershell
python -m pytest tests\test_guardrails.py -v
```

### Guardrail proof (manual)

With the server running (`uvicorn main:app --reload`) and a real `OPENAI_API_KEY` set:

**Empty question → 422**
```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "   "}'
```

**Oversized question → 422**
```powershell
$body = @{ question = ('a' * 4001) } | ConvertTo-Json
curl.exe -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d $body
```

**Rate limit → first 5 succeed, 6th returns 429**
```powershell
1..6 | ForEach-Object {
    curl.exe -s -o NUL -w "%{http_code}`n" -X POST http://127.0.0.1:8000/ask `
        -H "Content-Type: application/json" -d '{"question": "hi"}'
}
```

**Flagged content → 400**
```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" `
    -d '{"question": "I want to hurt someone"}'
```
Expect `detail` to name the moderation category (e.g. `violence`) that tripped the guardrail.

## Model cost writeup

`/ask` reports `cost_usd` per request, computed from the token usage OpenAI returns in the
streamed response (`stream_options={"include_usage": True}`) — **OpenAI does not return a
dollar cost itself**, only token counts, so the cost is calculated locally:

```
cost_usd = (prompt_tokens * price_per_prompt_token) + (completion_tokens * price_per_completion_token)
```

Prices live in `PRICING_PER_1M_TOKENS` in `main.py`, keyed by model:

| Model | Prompt ($ / 1M tokens) | Completion ($ / 1M tokens) |
|---|---|---|
| `gpt-4o-mini` | $0.15 | $0.60 |

Notes:

- These rates are **hardcoded** and must be updated by hand if OpenAI repricing happens or the
  `MODEL` constant changes to a model not yet in the table (in that case `cost_usd` is returned
  as `null` rather than silently guessing).
- Completion tokens are ~4x more expensive than prompt tokens for this model, which is why the
  cost/token-cap guardrail bounds `max_tokens` on the completion side — a runaway completion is
  the larger cost risk of the two.
- At current pricing, a typical short Q&A exchange (~100 prompt + ~150 completion tokens) costs
  roughly **$0.0001** per call — cheap per-call, but worth capping in aggregate via the rate
  limiter if this were exposed publicly.
- The Streamlit UI surfaces `cost_usd` in the "Token usage" expander under the response.

## Tests

```powershell
python -m pytest -v
```
