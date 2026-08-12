import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# USD per 1M tokens (OpenAI pricing as of writing).
PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
}

# --- Pinecone / embeddings config (all via env vars, no secrets in code) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Same model used at both ingest and query time so vectors stay comparable.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Chunking config, overridable via env vars without touching code.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Matches scoring below this are dropped as not relevant enough to use as context.
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.3"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

RAG_PROMPT_TEMPLATE = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used.

Context:
{retrieved_chunks}

Question: {question}"""

_pinecone_client: Pinecone | None = None

# --- Guardrail config ---
MAX_QUESTION_CHARS = 4000  # bounds prompt-side token/cost usage
MAX_COMPLETION_TOKENS = 500  # bounds completion-side token/cost usage
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60

# Per-client sliding window of request timestamps, for the in-memory rate limiter below.
_request_log: dict[str, deque[float]] = defaultdict(deque)


def check_rate_limit(client_id: str) -> None:
    now = time.monotonic()
    window = _request_log[client_id]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = round(RATE_LIMIT_WINDOW_SECONDS - (now - window[0]), 1)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {RATE_LIMIT_MAX_REQUESTS} requests per "
                f"{RATE_LIMIT_WINDOW_SECONDS}s. Retry after {retry_after}s."
            ),
        )
    window.append(now)


def get_pinecone_index():
    """Lazily builds the Pinecone client/index handle on first use, so `import main`
    still works without Pinecone configured (e.g. in tests, or before /ingest exists)."""
    global _pinecone_client
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is not set")
    if not PINECONE_INDEX_NAME:
        raise HTTPException(status_code=500, detail="PINECONE_INDEX_NAME is not set")

    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    return _pinecone_client.Index(PINECONE_INDEX_NAME)


def embed_text(text: str) -> list[float]:
    """Embeds text with EMBEDDING_MODEL. Call this at both ingest and query time
    so vectors come from the same model and stay comparable."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch version of embed_text for ingest, one API call for all chunks."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_grounding_messages(question: str, matches: list) -> list[dict]:
    """Fills RAG_PROMPT_TEMPLATE with the retrieved chunks and question."""
    if matches:
        context_block = "\n\n".join(
            f"[{m.metadata.get('document_id')}] {m.metadata.get('text')}" for m in matches
        )
    else:
        context_block = "(no relevant context found)"

    prompt = RAG_PROMPT_TEMPLATE.format(retrieved_chunks=context_block, question=question)
    return [{"role": "user", "content": prompt}]


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_be_valid(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question exceeds max length of {MAX_QUESTION_CHARS} characters")
        return v


class IngestRequest(BaseModel):
    text: str
    document_id: str
    source: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug/pinecone")
def debug_pinecone():
    """Confirms Pinecone is reachable and the configured index exists."""
    index = get_pinecone_index()
    try:
        stats = index.describe_index_stats()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone unreachable: {e}")

    return {
        "status": "ok",
        "index_name": PINECONE_INDEX_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
    }


@app.get("/debug/retrieve")
def debug_retrieve(q: str):
    """Embeds q and returns the top-5 nearest chunks from Pinecone. No LLM call —
    for verifying retrieval quality before wiring generation into /ask.
    curl example:
    curl.exe -s "http://127.0.0.1:8000/debug/retrieve?q=How+many+remote+days+are+allowed"
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    embedding = embed_text(q)
    index = get_pinecone_index()
    try:
        results = index.query(vector=embedding, top_k=5, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone unreachable: {e}")

    return {
        "query": q,
        "min_score": MIN_SCORE_THRESHOLD,
        "matches": [
            {
                "score": match.score,
                "document_id": match.metadata.get("document_id"),
                "chunk_index": match.metadata.get("chunk_index"),
                "source": match.metadata.get("source"),
                "text": match.metadata.get("text"),
            }
            for match in results.matches
            if match.score >= MIN_SCORE_THRESHOLD
        ],
    }


@app.post("/ingest")
def ingest(body: IngestRequest):
    # curl example:
    # curl.exe -s -X POST http://127.0.0.1:8000/ingest \
    #   -H "Content-Type: application/json" \
    #   -d '{"text": "Some document text...", "document_id": "doc-1", "source": "handbook.pdf"}'
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not body.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id must not be empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(body.text)

    embeddings = embed_texts(chunks)

    vectors = [
        {
            "id": f"{body.document_id}-{i}",
            "values": embedding,
            "metadata": {
                "document_id": body.document_id,
                "chunk_index": i,
                "source": body.source or "",
                "text": chunk,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    index = get_pinecone_index()
    try:
        index.upsert(vectors=vectors)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone unreachable: {e}")

    return {
        "document_id": body.document_id,
        "chunks_indexed": len(chunks),
        "status": "ok",
    }


@app.post("/ask")
def ask(body: AskRequest, request: Request):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    check_rate_limit(request.client.host if request.client else "unknown")

    moderation = client.moderations.create(input=body.question)
    if moderation.results[0].flagged:
        flagged_categories = [
            category
            for category, flagged in moderation.results[0].categories.model_dump().items()
            if flagged
        ]
        raise HTTPException(
            status_code=400,
            detail=f"Question flagged by content moderation: {', '.join(flagged_categories)}",
        )

    # --- Retrieval ---
    question_embedding = embed_text(body.question)
    index = get_pinecone_index()
    try:
        results = index.query(vector=question_embedding, top_k=RETRIEVAL_TOP_K, include_metadata=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone unreachable: {e}")
    matches = [m for m in results.matches if m.score >= MIN_SCORE_THRESHOLD]

    # --- Generation (same streaming/usage/cost path as Session 1) ---
    start = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    usage = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=build_grounding_messages(body.question, matches),
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=MAX_COMPLETION_TOKENS,
    )

    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks.append(delta)

    total_latency = time.perf_counter() - start
    answer = "".join(chunks)
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None

    cost_usd = None
    pricing = PRICING_PER_1M_TOKENS.get(MODEL)
    if pricing and usage:
        cost_usd = round(
            (prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"])
            / 1_000_000,
            6,
        )

    # Throughput after the first token arrives (generation speed).
    generation_time = (total_latency - ttft) if ttft is not None else total_latency
    tokens_per_sec = (
        round(completion_tokens / generation_time, 2)
        if completion_tokens and generation_time > 0
        else None
    )

    return {
        "answer": answer,
        "model": MODEL,
        "retrieved_chunks": [
            {
                "chunk_id": m.id,
                "document_id": m.metadata.get("document_id"),
                "score": m.score,
            }
            for m in matches
        ],
        "metrics": {
            "time_to_first_token_sec": round(ttft, 3) if ttft is not None else None,
            "total_latency_sec": round(total_latency, 3),
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.total_tokens if usage else None,
            "cost_usd": cost_usd,
        },
    }
