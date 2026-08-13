import os

import requests
import streamlit as st

REFUSAL_MARKER = "don't have enough information"

st.set_page_config(page_title="RAG Console", page_icon="📚")

st.sidebar.header("API")
api_base_url = st.sidebar.text_input(
    "API base URL",
    value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
    help="Your FastAPI service, e.g. https://ai-eng-bootcamp-20lm.onrender.com",
).rstrip("/")

st.title("RAG Console")
st.caption("Talks to your FastAPI /ingest and /ask endpoints. No RAG logic runs here.")

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ask_tab:
    question = st.text_area("Question", placeholder="Ask something…", height=100, key="question")
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Thinking…"):
                try:
                    response = requests.post(
                        f"{api_base_url}/ask",
                        json={"question": question.strip()},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer", "")
                    is_refusal = REFUSAL_MARKER in answer.lower()

                    st.subheader("Response")
                    if is_refusal:
                        st.warning(f"🚫 Refused — insufficient context\n\n{answer}")
                    else:
                        st.success(answer)
                    st.caption(f"Model: {data.get('model')}")

                    chunks = data.get("retrieved_chunks", [])
                    st.subheader(f"Retrieved chunks ({len(chunks)})")
                    if chunks:
                        st.table(
                            [
                                {
                                    "document_id": c.get("document_id"),
                                    "chunk_id": c.get("chunk_id"),
                                    "score": round(c.get("score", 0), 3),
                                }
                                for c in chunks
                            ]
                        )
                    else:
                        st.caption("No chunks retrieved.")

                    metrics = data.get("metrics", {})
                    with st.expander("Token usage & cost"):
                        st.write(
                            {
                                "prompt_tokens": metrics.get("prompt_tokens"),
                                "completion_tokens": metrics.get("completion_tokens"),
                                "total_tokens": metrics.get("total_tokens"),
                            }
                        )
                        cost_usd = metrics.get("cost_usd")
                        st.caption(f"Cost: ${cost_usd:.6f}" if cost_usd is not None else "Cost: —")
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not reach the API at {api_base_url}. Is it running?")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.status_code} — {e.response.text}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

with ingest_tab:
    text = st.text_area("Text", placeholder="Paste document text…", height=200, key="ingest_text")
    document_id = st.text_input("document_id", placeholder="e.g. POL-101", key="ingest_doc_id")
    source = st.text_input("source (optional)", placeholder="e.g. handbook.pdf", key="ingest_source")
    ingest = st.button("Ingest", type="primary")

    if ingest:
        if not text.strip():
            st.warning("Paste some text first.")
        elif not document_id.strip():
            st.warning("document_id is required.")
        else:
            with st.spinner("Ingesting…"):
                try:
                    payload = {"text": text.strip(), "document_id": document_id.strip()}
                    if source.strip():
                        payload["source"] = source.strip()

                    response = requests.post(f"{api_base_url}/ingest", json=payload, timeout=60)
                    response.raise_for_status()
                    data = response.json()

                    st.success(
                        f"Ingested `{data['document_id']}` — {data['chunks_indexed']} chunks indexed."
                    )
                    st.json(data)
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not reach the API at {api_base_url}. Is it running?")
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.status_code} — {e.response.text}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
