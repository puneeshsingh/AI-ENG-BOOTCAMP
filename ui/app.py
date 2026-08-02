import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Ask", page_icon="💬")
st.title("Ask")
st.caption("Talks to your FastAPI /ask endpoint")

question = st.text_area("Question", placeholder="Ask something…", height=100)
ask = st.button("Ask", type="primary")

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question.strip()},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                st.subheader("Response")
                st.markdown(data["answer"])
                st.caption(f"Model: {data['model']}")

                metrics = data.get("metrics", {})
                st.subheader("Latency metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Time to first token",
                    f"{metrics.get('time_to_first_token_sec')} s"
                    if metrics.get("time_to_first_token_sec") is not None
                    else "—",
                )
                col2.metric(
                    "Total latency",
                    f"{metrics.get('total_latency_sec')} s"
                    if metrics.get("total_latency_sec") is not None
                    else "—",
                )
                col3.metric(
                    "Tokens / sec",
                    metrics.get("tokens_per_sec")
                    if metrics.get("tokens_per_sec") is not None
                    else "—",
                )

                with st.expander("Token usage"):
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
                st.error("Could not reach the API. Is uvicorn running on port 8000?")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
