from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from .config import load_settings
from .data import dataset_insights, load_reviews
from .evaluation import RUBRIC, empty_rubric_results
from .llm import LLMClient
from .prompts import get_prompt, load_prompts, prompt_metadata
from .runner import run_prompt


st.set_page_config(page_title="Customer Review Prompt Engineering", layout="wide")


@st.cache_data(show_spinner=False)
def cached_reviews(path: str, sample_size: int) -> pd.DataFrame:
    return load_reviews(Path(path), sample_size=sample_size)


def main() -> None:
    settings = load_settings()
    st.title("Customer Review Prompt Engineering")
    st.caption("Amazon Fine Food Reviews: design, test, evaluate, improve, and compare prompt versions.")

    with st.sidebar:
        st.header("Controls")
        sample_size = st.slider("Dataset rows to load", 100, 5000, 1000, step=100)
        task = st.selectbox("Task", ["data_analysis", "summarization", "classification", "content_generation"])
        version = st.radio("Prompt version", ["v1", "v2", "v3"], horizontal=True)
        input_mode = st.radio("Review input", ["Dataset rows", "Custom text"])

    df = cached_reviews(str(settings["data_path"]), sample_size)
    insights = dataset_insights(df)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviews Loaded", f"{insights['review_count']:,}")
    metric_cols[1].metric("Average Rating", insights["average_score"])
    metric_cols[2].metric("Positive", insights["label_distribution"].get("Positive", 0))
    metric_cols[3].metric("Negative", insights["label_distribution"].get("Negative", 0))

    chart_col, table_col = st.columns([1, 1])
    with chart_col:
        score_df = pd.DataFrame({"score": list(insights["score_distribution"].keys()), "count": list(insights["score_distribution"].values())})
        st.plotly_chart(px.bar(score_df, x="score", y="count", title="Rating Distribution"), use_container_width=True)
    with table_col:
        st.dataframe(df[["Score", "reference_label", "Summary", "Text"]].head(12), use_container_width=True, height=340)

    if input_mode == "Dataset rows":
        count = st.slider("Reviews sent to prompt", 1, 10, 5)
        selected = df.head(count)
        prompt_reviews = selected
    else:
        custom = st.text_area("Paste one or more reviews", height=180)
        prompt_reviews = custom.strip()

    meta = prompt_metadata(task, version)
    st.subheader(f"{task.replace('_', ' ').title()} | {version.upper()}: {meta['name']}")
    st.write(meta["goal"])

    prompt_preview = get_prompt(task, version, {"reviews": "[selected reviews]", "review": "[selected review]", "dataset_insights": str(insights)})
    with st.expander("Prompt preview"):
        st.code(prompt_preview, language="text")

    run_cols = st.columns([1, 1])
    with run_cols[0]:
        if st.button("Run selected prompt", type="primary"):
            try:
                client = LLMClient(settings["api_key"], settings["model"])
                response = run_prompt(task, version, prompt_reviews, client, insights)
                st.session_state["last_response"] = response
            except Exception as exc:
                st.error(str(exc))
    with run_cols[1]:
        compare = st.button("Compare V1, V2, V3")

    if "last_response" in st.session_state:
        st.subheader("AI Response")
        st.markdown(st.session_state["last_response"])

    if compare:
        try:
            client = LLMClient(settings["api_key"], settings["model"])
            cols = st.columns(3)
            for i, v in enumerate(["v1", "v2", "v3"]):
                with cols[i]:
                    st.markdown(f"**{v.upper()}**")
                    st.markdown(run_prompt(task, v, prompt_reviews, client, insights))
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Evaluation Rubric")
    if task == "classification":
        st.info("Classification metrics are calculated in `experiments/run_classification_eval.py` using rating-derived reference labels.")
    else:
        st.dataframe(empty_rubric_results(task, version), use_container_width=True)
        st.caption("Rubric scores are intentionally blank until the prompts are actually run and reviewed.")


if __name__ == "__main__":
    main()
