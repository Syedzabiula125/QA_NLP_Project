"""Professional Streamlit UI for the Hugging Face QA service."""
from __future__ import annotations

import io

import streamlit as st

import config
from qa_pipeline import QuestionAnsweringService
from utils import create_visualizations, parse_batch_questions, read_context_file, results_to_frame, save_results_csv

st.set_page_config(page_title="NLP Question Answering", page_icon="💬", layout="wide")
SERVICE_CACHE_VERSION = "qa-pipeline-input-v2"


@st.cache_resource(show_spinner=False)
def get_service(
    model_name: str, token: str, cache_version: str
) -> QuestionAnsweringService:
    """Keep a loaded model service across Streamlit reruns."""
    # Include a source-version key so a pipeline-behaviour update invalidates
    # previously cached service instances after Streamlit hot reload.
    del cache_version
    return QuestionAnsweringService(model_name, token=token or None)


def initialise_state() -> None:
    """Create session data used by the controls."""
    if "results" not in st.session_state:
        st.session_state.results = []
    if "context" not in st.session_state:
        try:
            st.session_state.context = read_context_file()
        except (FileNotFoundError, ValueError):
            st.session_state.context = ""


def main() -> None:
    initialise_state()
    st.title("NLP Question Answering System")
    st.caption("Hugging Face Transformers Pipeline API • extractive QA with a Llama-safe fallback")
    with st.sidebar:
        st.header("Configuration")
        model_name = st.text_input("Model name", value=config.MODEL_NAME)
        hf_token = st.text_input(
            "Hugging Face token (optional)", type="password",
            help="Required only if you want to use an approved gated model. It is not logged."
        )
        service = get_service(model_name, hf_token, SERVICE_CACHE_VERSION)
        st.subheader("Model information")
        st.json(service.info())
        if st.button("Load model", use_container_width=True):
            try:
                with st.spinner("Loading model; first download can take time..."):
                    service.load()
                st.success(f"Loaded: {service.mode}")
                if service.fallback_reason:
                    st.warning(
                        f"Requested model unavailable. Active model: "
                        f"{service.active_model_name}. {service.fallback_reason}"
                    )
            except RuntimeError as exc:
                st.error(str(exc))
        if st.button("Save loaded model", use_container_width=True):
            try:
                st.success(f"Saved to {service.save_model()}")
            except RuntimeError as exc:
                st.error(str(exc))
        if st.button("Clear results", use_container_width=True):
            st.session_state.results = []
            st.rerun()

    context = st.text_area("Context", key="context", height=220, placeholder="Paste or write the source text here.")
    single_tab, batch_tab, results_tab, charts_tab = st.tabs(["Single question", "Batch questions", "Results", "Visualizations"])
    with single_tab:
        question = st.text_input("Question", placeholder="What would you like to know?")
        if st.button("Answer question", type="primary"):
            try:
                with st.spinner("Running inference..."):
                    result = service.answer(question, context)
                st.session_state.results.append(result)
                left, middle, right = st.columns(3)
                left.metric("Answer", result["answer"] or "No answer")
                middle.metric("Confidence", "N/A" if result["confidence"] is None else f"{result['confidence']:.2%}")
                right.metric("Processing time", f"{result['processing_time_seconds']:.4f} s")
                if result["error"]:
                    st.info(result["error"])
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    with batch_tab:
        raw_questions = st.text_area("One question per line", height=160, placeholder="Question one?\nQuestion two?")
        if st.button("Answer batch"):
            try:
                with st.spinner("Running batch inference..."):
                    batch_results = service.answer_batch(parse_batch_questions(raw_questions), context)
                st.session_state.results.extend(batch_results)
                st.success(f"Processed {len(batch_results)} question(s).")
                st.dataframe(results_to_frame(batch_results), use_container_width=True)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    with results_tab:
        if st.session_state.results:
            frame = results_to_frame(st.session_state.results)
            st.dataframe(frame, use_container_width=True, hide_index=True)
            csv_bytes = frame.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, "qa_results.csv", "text/csv")
            if st.button("Save CSV to outputs/"):
                st.success(f"Saved: {save_results_csv(st.session_state.results)}")
        else:
            st.info("Answers will appear here after inference.")
    with charts_tab:
        if st.session_state.results:
            for name, figure in create_visualizations(st.session_state.results).items():
                st.pyplot(figure, use_container_width=True)
        else:
            st.info("Run at least one question to generate charts.")


if __name__ == "__main__":
    main()
