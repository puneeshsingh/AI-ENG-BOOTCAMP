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
PINECONE_API_KEY=pcsk-...
PINECONE_INDEX_NAME=your-index-name
```

`PINECONE_API_KEY`/`PINECONE_INDEX_NAME` are only required for the Pinecone-backed
endpoints (currently `GET /debug/pinecone`); `/ask` and `/health` work without them.
The Pinecone index must already exist (created via the Pinecone console or API) with
dimension `1536` and metric `cosine`, matching `text-embedding-3-small`
(`EMBEDDING_MODEL` in `main.py`) — this is the same model used at both ingest and
query time so vectors stay comparable.

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

### `GET /debug/pinecone`

Confirms Pinecone is reachable and reports basic index stats. Requires
`PINECONE_API_KEY` and `PINECONE_INDEX_NAME` to be set; returns `500` if either is
missing, `502` if Pinecone can't be reached or the index doesn't exist.

```json
{
  "status": "ok",
  "index_name": "your-index-name",
  "embedding_model": "text-embedding-3-small",
  "vector_count": 0,
  "dimension": 1536
}
```

### `POST /ingest`

Chunks text with `RecursiveCharacterTextSplitter` (`CHUNK_SIZE`/`CHUNK_OVERLAP` env
vars, default 800/100), embeds each chunk with `text-embedding-3-small`, and upserts
into Pinecone. Vector IDs are `{document_id}-{chunk_index}`, so re-ingesting the same
`document_id` overwrites its previous chunks rather than duplicating them.

```json
{ "text": "...", "document_id": "doc-1", "source": "handbook.pdf" }
```

`source` is optional. Returns `400` if `text` or `document_id` is blank/whitespace-only.

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" `
  -d '{"text": "Some document text...", "document_id": "doc-1", "source": "handbook.pdf"}'
```

Response:

```json
{ "document_id": "doc-1", "chunks_indexed": 4, "status": "ok" }
```

Each vector's metadata is `{"document_id", "chunk_index", "source", "text"}` — the
chunk text is stored alongside the IDs so retrieval can return matched content
directly, without a separate lookup.

### `GET /debug/retrieve?q=...`

Embeds `q` with the same `text-embedding-3-small` model used at ingest, queries
Pinecone for the top 5 nearest chunks, and drops any scoring below
`MIN_SCORE_THRESHOLD` (env var, default `0.3`) so unrelated chunks don't get returned
just to fill out top-k. Does **not** call the LLM — for verifying retrieval quality
before wiring generation into `/ask`. Returns `400` if `q` is blank.

```powershell
curl.exe -s "http://127.0.0.1:8000/debug/retrieve?q=How+many+remote+days+are+allowed"
```

```json
{
  "query": "How many remote days are allowed",
  "min_score": 0.3,
  "matches": [
    {
      "score": 0.579,
      "document_id": "remote-work-policy",
      "chunk_index": 0,
      "source": "",
      "text": "Remote Work Policy. Employees may work remotely up to 3 days per week..."
    }
  ]
}
```

### `POST /ask`

Retrieval-augmented: embeds the question, retrieves up to `RETRIEVAL_TOP_K` (default 5)
chunks from Pinecone above `MIN_SCORE_THRESHOLD`, and answers strictly from that
context using `RAG_PROMPT_TEMPLATE` in `main.py`:

```
Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}
```

```json
{ "question": "How many remote days are allowed?" }
```

Returns the answer, the chunks used to ground it, plus latency/token/cost metrics
(unchanged formula from Session 1 — completion cost only, embedding cost isn't
included in `cost_usd`):

```json
{
  "answer": "Employees may work remotely up to 3 days per week, subject to manager approval. (document_id: remote-work-policy)",
  "model": "gpt-4o-mini",
  "retrieved_chunks": [
    { "chunk_id": "remote-work-policy-0", "document_id": "remote-work-policy", "score": 0.590 }
  ],
  "metrics": {
    "time_to_first_token_sec": 0.562,
    "total_latency_sec": 0.763,
    "tokens_per_sec": 124.03,
    "prompt_tokens": 323,
    "completion_tokens": 25,
    "total_tokens": 348,
    "cost_usd": 0.000063
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
