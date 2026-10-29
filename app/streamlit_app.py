"""Offline Streamlit demo for ECBias."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ecbias.viz.figures import LABELS, PALETTE
from ecbias.viz.live import analyze_text
from ecbias.viz.power import simulate_power

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    file_path = ROOT / path
    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _read_text(relative_path: str) -> str:
    path = ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else "Summary is not available yet."


def overview() -> None:
    st.header("Overview")
    exchanges = load_table("data/interim/exchanges_enriched.parquet")
    quality = load_table("results/tables/data_quality.csv")
    st.metric("Exchanges", len(exchanges))
    st.dataframe(quality, use_container_width=True)
    st.info(
        "Limitations: gender is inferred perceived gender from names; AQS is a proxy; "
        "data are observational associations, not proof of bias."
    )


def argument_explorer() -> None:
    st.header("Argument explorer")
    sentences = load_table("data/processed/sentences_labeled.parquet")
    exchanges = load_table("data/interim/exchanges_enriched.parquet")
    merged = sentences.merge(
        exchanges[["exchange_id", "call_id", "analyst_gender"]], on="exchange_id", how="left"
    )
    gender = st.selectbox("Analyst gender", ["All", "F", "M", "Unknown"])
    if gender != "All":
        merged = merged[merged["analyst_gender"].eq(gender)]
    for label in LABELS:
        st.markdown(
            f"<span style='background:{PALETTE[label]};padding:4px'>{label}</span>",
            unsafe_allow_html=True,
        )
    st.dataframe(
        merged[["call_id", "exchange_id", "side", "sent_idx", "label", "text"]],
        use_container_width=True,
    )


def quality_response() -> None:
    st.header("Quality and response metrics")
    features = load_table("data/processed/features.parquet")
    metric = st.selectbox(
        "Metric", ["AQS_q", "AQS_a", "relevance_tfidf", "answer_to_question_ratio"]
    )
    if metric in features:
        st.bar_chart(features[metric].dropna().value_counts().sort_index())
    st.dataframe(features.sort_values(metric, ascending=False), use_container_width=True)


def bias_results() -> None:
    st.header("Bias results")
    image = ROOT / "results/figures/04_primary_effect_forest.png"
    if image.exists():
        st.image(str(image))
    table = ROOT / "results/tables/tests_primary.csv"
    if table.exists():
        st.dataframe(load_table("results/tables/tests_primary.csv"), use_container_width=True)
    st.markdown(_read_text("results/SUMMARY.md"))


def live_analyser() -> None:
    st.header("Live analyser")
    question = st.text_area("Question", "What drove revenue growth this quarter?")
    answer = st.text_area("Answer", "Revenue grew 10 percent because demand improved.")
    if st.button("Analyse"):
        result = analyze_text(question, answer)
        st.caption(f"Completed in {result['elapsed_ms']:.1f} ms")
        st.dataframe(result["sentences"], use_container_width=True)
        score = result["scores"]
        components = {
            key: value for key, value in score.items() if key.endswith("_q") and key != "AQS_q"
        }
        st.bar_chart(pd.Series(components))
        st.json({key: value for key, value in result["response"].items() if key != "exchange_id"})
        st.caption(
            "Higher numeric specificity and structure raise the rubric score; hedging and "
            "low focus can lower it."
        )


def power_page() -> None:
    st.header("Power simulator")
    effect = st.slider("True effect size (Cohen's d)", -1.0, 1.0, 0.2, 0.05)
    n_female = st.slider("Female analyst questions", 10, 500, 60)
    n_male = st.slider("Male analyst questions", 10, 500, 120)
    studies = st.slider("Simulated studies", 100, 5000, 1000, 100)
    p_values, power = simulate_power(effect, n_female, n_male, studies)
    st.metric("Share detecting the effect", f"{power:.1%}")
    st.caption("Small samples often cannot detect small effects reliably.")
    st.line_chart(pd.Series(np.sort(p_values), name="p-value"))


def main() -> None:
    st.set_page_config(page_title="ECBias", layout="wide")
    st.sidebar.title("ECBias")
    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Argument explorer",
            "Quality and response metrics",
            "Bias results",
            "Live analyser",
            "Power simulator",
        ],
    )
    st.sidebar.markdown("[Methods and hypotheses](../docs/HYPOTHESES.md)")
    {
        "Overview": overview,
        "Argument explorer": argument_explorer,
        "Quality and response metrics": quality_response,
        "Bias results": bias_results,
        "Live analyser": live_analyser,
        "Power simulator": power_page,
    }[page]()


if __name__ == "__main__":
    main()
