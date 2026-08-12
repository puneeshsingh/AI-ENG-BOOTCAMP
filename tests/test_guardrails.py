from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def reset_guardrail_state(monkeypatch):
    """Each test gets a clean rate-limit window and a real API key set."""
    main._request_log.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield
    main._request_log.clear()


@pytest.fixture(autouse=True)
def mock_retrieval(monkeypatch):
    """/ask now embeds the question and queries Pinecone before generating. Mock both
    so the suite stays offline by default; tests that care about retrieval override
    main.get_pinecone_index directly."""
    monkeypatch.setattr(main, "embed_text", lambda text: [0.0] * main.EMBEDDING_DIMENSIONS)
    fake_index = SimpleNamespace(query=lambda **kwargs: SimpleNamespace(matches=[]))
    monkeypatch.setattr(main, "get_pinecone_index", lambda: fake_index)
    yield


@pytest.fixture
def client():
    return TestClient(main.app)


def make_moderation_response(flagged: bool, categories: dict[str, bool] | None = None):
    categories = categories or {}
    return SimpleNamespace(
        results=[
            SimpleNamespace(
                flagged=flagged,
                categories=SimpleNamespace(model_dump=lambda: categories),
            )
        ]
    )


def make_stream(text: str, prompt_tokens: int, completion_tokens: int):
    """Mimics the chunks yielded by client.chat.completions.create(stream=True)."""
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        ),
    ]
    return iter(chunks)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_empty_question_rejected(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_oversized_question_rejected(client):
    oversized = "a" * (main.MAX_QUESTION_CHARS + 1)
    response = client.post("/ask", json={"question": oversized})
    assert response.status_code == 422


def test_missing_api_key_rejected(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 500


def test_moderation_blocks_flagged_content(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations,
        "create",
        lambda input: make_moderation_response(True, {"harassment": True, "violence": False}),
    )
    response = client.post("/ask", json={"question": "something disallowed"})
    assert response.status_code == 400
    assert "harassment" in response.json()["detail"]


def test_successful_response_includes_cost_usd(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations, "create", lambda input: make_moderation_response(False)
    )
    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        lambda **kwargs: make_stream("hi there", prompt_tokens=100, completion_tokens=50),
    )

    response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "hi there"

    expected_cost = round((100 * 0.15 + 50 * 0.60) / 1_000_000, 6)
    assert body["metrics"]["cost_usd"] == expected_cost
    assert body["metrics"]["prompt_tokens"] == 100
    assert body["metrics"]["completion_tokens"] == 50


def test_completion_request_caps_max_tokens(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations, "create", lambda input: make_moderation_response(False)
    )
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return make_stream("ok", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(main.client.chat.completions, "create", fake_create)

    client.post("/ask", json={"question": "hello"})
    assert captured_kwargs["max_tokens"] == main.MAX_COMPLETION_TOKENS


def make_match(chunk_id, document_id, score, text):
    return SimpleNamespace(id=chunk_id, score=score, metadata={"document_id": document_id, "text": text})


def test_ask_includes_retrieved_chunks_and_grounds_the_prompt(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations, "create", lambda input: make_moderation_response(False)
    )
    match = make_match("remote-work-policy-0", "remote-work-policy", 0.58, "Up to 3 remote days per week.")
    monkeypatch.setattr(
        main, "get_pinecone_index", lambda: SimpleNamespace(query=lambda **kw: SimpleNamespace(matches=[match]))
    )
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return make_stream("Up to 3 days [remote-work-policy]", prompt_tokens=50, completion_tokens=10)

    monkeypatch.setattr(main.client.chat.completions, "create", fake_create)

    response = client.post("/ask", json={"question": "How many remote days are allowed?"})
    assert response.status_code == 200
    body = response.json()

    assert body["retrieved_chunks"] == [
        {"chunk_id": "remote-work-policy-0", "document_id": "remote-work-policy", "score": 0.58}
    ]
    # the retrieved chunk text must actually reach the model as grounding context
    user_message = captured_kwargs["messages"][0]["content"]
    assert "Up to 3 remote days per week." in user_message
    assert "How many remote days are allowed?" in user_message


def test_ask_with_no_matches_still_answers_with_empty_context(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations, "create", lambda input: make_moderation_response(False)
    )
    # default mock_retrieval fixture already returns no matches
    captured_kwargs = {}

    def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        return make_stream(
            "I don't have enough information to answer that.",
            prompt_tokens=40,
            completion_tokens=15,
        )

    monkeypatch.setattr(main.client.chat.completions, "create", fake_create)

    response = client.post("/ask", json={"question": "What is the meaning of life?"})
    assert response.status_code == 200
    body = response.json()

    assert body["retrieved_chunks"] == []
    assert "(no relevant context found)" in captured_kwargs["messages"][0]["content"]


def test_rate_limit_exceeded(client, monkeypatch):
    monkeypatch.setattr(
        main.client.moderations, "create", lambda input: make_moderation_response(False)
    )
    monkeypatch.setattr(
        main.client.chat.completions,
        "create",
        lambda **kwargs: make_stream("ok", prompt_tokens=1, completion_tokens=1),
    )

    for _ in range(main.RATE_LIMIT_MAX_REQUESTS):
        response = client.post("/ask", json={"question": "hello"})
        assert response.status_code == 200

    response = client.post("/ask", json={"question": "hello"})
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
