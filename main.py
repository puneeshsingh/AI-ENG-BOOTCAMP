import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, field_validator

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# USD per 1M tokens (OpenAI pricing as of writing).
PRICING_PER_1M_TOKENS = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
}

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


@app.get("/health")
def health():
    return {"status": "ok"}


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

    start = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    usage = None

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": body.question}],
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
