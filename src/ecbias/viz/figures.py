"""Publication-oriented offline figures for ECBias outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {
    "F": "#CC79A7",
    "M": "#0072B2",
    "Unknown": "#999999",
    "CLAIM": "#E69F00",
    "EVIDENCE": "#009E73",
    "REBUTTAL": "#D55E00",
    "REQUEST": "#56B4E9",
    "OTHER": "#999999",
}
LABELS = ["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "axes.titleweight": "bold"})


def _finish(fig: plt.Figure, name: str, caption: str, output_dir: str | Path | None) -> plt.Figure:
    if output_dir is None:
        return fig
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    caption_path = directory / "captions.csv"
    row = pd.DataFrame([{"figure": name, "alt_text": caption}])
    if caption_path.exists():
        existing = pd.read_csv(caption_path)
        row = pd.concat([existing[existing["figure"].ne(name)], row], ignore_index=True)
    row.to_csv(caption_path, index=False)
    return fig


def _empty(title: str, message: str) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_title(title)
    axis.axis("off")
    return fig


def _empty_saved(title: str, message: str, name: str, caption: str, output_dir) -> plt.Figure:
    return _finish(_empty(title, message), name, caption, output_dir)


def coverage_figure(exchanges: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    calls = exchanges["call_id"].nunique() if "call_id" in exchanges else 0
    axes[0].bar(["Calls", "Exchanges"], [calls, len(exchanges)], color=["#0072B2", "#E69F00"])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Corpus coverage")
    shares = []
    for column, population in [
        ("analyst_gender", "Analyst"),
        ("first_responder_gender", "Executive"),
    ]:
        if column in exchanges:
            values = exchanges[column].fillna("Unknown").value_counts(normalize=True)
            shares.append([values.get(gender, 0) * 100 for gender in ["F", "M", "Unknown"]])
        else:
            shares.append([0, 0, 0])
    axes[1].bar(["Analyst", "Executive"], [row[0] for row in shares], label="F", color=PALETTE["F"])
    axes[1].bar(
        ["Analyst", "Executive"],
        [row[1] for row in shares],
        bottom=[row[0] for row in shares],
        label="M",
        color=PALETTE["M"],
    )
    axes[1].bar(
        ["Analyst", "Executive"],
        [row[2] for row in shares],
        bottom=[row[0] + row[1] for row in shares],
        label="Unknown",
        color=PALETTE["Unknown"],
    )
    axes[1].set_ylabel("Share (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Perceived gender coverage")
    axes[1].legend()
    return _finish(
        fig,
        "01_data_coverage",
        "Corpus contains calls and exchanges; gender shares include Unknown.",
        output_dir,
    )


def argument_mix_figure(
    sentences: pd.DataFrame, exchanges: pd.DataFrame, output_dir=None
) -> plt.Figure:
    _style()
    merged = sentences.merge(
        exchanges[["exchange_id", "analyst_gender"]], on="exchange_id", how="left"
    )
    counts = merged.groupby(["side", "analyst_gender", "label"]).size().unstack(fill_value=0)
    if counts.empty:
        return _empty_saved(
            "Argument component mix",
            "No labelled sentences available.",
            "02_argument_component_mix",
            "No labelled argument sentences were available.",
            output_dir,
        )
    counts = counts.reindex(columns=LABELS, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(shares))
    bottom = np.zeros(len(shares))
    for label in LABELS:
        values = shares[label].to_numpy()
        axis.bar(x, values, bottom=bottom, label=label, color=PALETTE[label])
        bottom += values
    axis.set_xticks(x, [f"{side}\n{gender}" for side, gender in shares.index])
    axis.set_ylabel("Share of sentences")
    axis.set_title("Argument components differ in composition by side and analyst gender")
    axis.legend(ncol=3)
    return _finish(
        fig,
        "02_argument_component_mix",
        "Stacked sentence-label shares are shown by exchange side and analyst gender.",
        output_dir,
    )


def aqs_distribution_figure(features: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    if "AQS_q" not in features or "analyst_gender" not in features:
        return _empty_saved(
            "Question quality by analyst gender",
            "AQS_q or analyst gender is unavailable.",
            "03_aqs_q_distribution",
            "Question AQS or analyst gender was unavailable.",
            output_dir,
        )
    data = features.dropna(subset=["AQS_q", "analyst_gender"]).copy()
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=data,
        x="analyst_gender",
        y="AQS_q",
        order=["F", "M"],
        palette=PALETTE,
        cut=0,
        inner=None,
        ax=axis,
    )
    sns.boxplot(
        data=data,
        x="analyst_gender",
        y="AQS_q",
        order=["F", "M"],
        width=0.2,
        color="white",
        ax=axis,
    )
    for position, gender in enumerate(["F", "M"]):
        axis.text(
            position,
            axis.get_ylim()[1],
            f"n={int(data['analyst_gender'].eq(gender).sum())}",
            ha="center",
            va="top",
        )
    axis.set_title("Question quality distribution by perceived analyst gender")
    axis.set_xlabel("Perceived analyst gender")
    axis.set_ylabel("AQS_q (0–100)")
    return _finish(
        fig,
        "03_aqs_q_distribution",
        "Question AQS distributions are displayed by perceived analyst gender with sample sizes.",
        output_dir,
    )


def forest_figure(tests: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    required = {"outcome", "effect_size", "ci_low", "ci_high"}
    if not required.issubset(tests.columns) or tests.empty:
        return _empty_saved(
            "Primary outcome effect sizes",
            "Primary test results are not available.",
            "04_primary_effect_forest",
            "Primary test results were not available.",
            output_dir,
        )
    data = tests.copy().sort_values("effect_size")
    fig, axis = plt.subplots(figsize=(9, max(4, len(data) * 0.45)))
    y = np.arange(len(data))
    axis.errorbar(
        data["effect_size"],
        y,
        xerr=[data["effect_size"] - data["ci_low"], data["ci_high"] - data["effect_size"]],
        fmt="o",
        color="#0072B2",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=1)
    labels = data["outcome"].astype(str)
    if "p_adj" in data:
        labels = labels + data["p_adj"].map(lambda p: " *" if pd.notna(p) and p < 0.05 else "")
    axis.set_yticks(y, labels)
    axis.set_xlabel("Standardised effect size (95% bootstrap CI)")
    axis.set_title("Estimated gender differences in primary outcomes")
    return _finish(
        fig,
        "04_primary_effect_forest",
        "Forest plot shows standardised effect estimates and bootstrap confidence intervals; "
        "stars mark BH-adjusted p<0.05.",
        output_dir,
    )


def answer_quality_heatmap(features: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    required = {"analyst_gender", "first_responder_gender", "AQS_a"}
    if not required.issubset(features.columns):
        return _empty_saved(
            "Answer quality by gender cell",
            "Answer quality or gender cell data is unavailable.",
            "05_answer_quality_heatmap",
            "Answer quality or gender cell data was unavailable.",
            output_dir,
        )
    grouped = (
        features.dropna(subset=["AQS_a"])
        .groupby(["analyst_gender", "first_responder_gender"])["AQS_a"]
        .agg(["mean", "size"])
        .reindex(pd.MultiIndex.from_product([["F", "M"], ["F", "M"]]))
    )
    table = grouped["mean"].unstack()
    annotations = grouped.apply(
        lambda row: (
            f"{row['mean']:.1f}\n(n={int(row['size'])})" if pd.notna(row["mean"]) else "n/a"
        ),
        axis=1,
    ).unstack()
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        table,
        annot=annotations,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Mean AQS_a"},
        ax=axis,
    )
    axis.set_xlabel("Perceived executive gender")
    axis.set_ylabel("Perceived analyst gender")
    axis.set_title("Mean answer quality by analyst–executive gender cell")
    return _finish(
        fig,
        "05_answer_quality_heatmap",
        "Mean answer AQS is shown for each analyst–executive perceived-gender cell "
        "with cell counts.",
        output_dir,
    )


def answer_ratio_order_figure(features: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    required = {"order_norm", "answer_to_question_ratio", "analyst_gender"}
    if not required.issubset(features.columns):
        return _empty_saved(
            "Answer length ratio by question order",
            "Order or ratio data is unavailable.",
            "06_answer_ratio_by_order",
            "Order or answer-ratio data was unavailable.",
            output_dir,
        )
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=features,
        x="order_norm",
        y="answer_to_question_ratio",
        hue="analyst_gender",
        palette=PALETTE,
        alpha=0.7,
        ax=axis,
    )
    axis.set_xlabel("Normalised question order within call")
    axis.set_ylabel("Answer/question word ratio")
    axis.set_title("Answer length ratio across question order")
    return _finish(
        fig,
        "06_answer_ratio_by_order",
        "Answer-to-question word ratios are shown across normalised call order by analyst gender.",
        output_dir,
    )


def robustness_figure(robustness: pd.DataFrame, output_dir=None) -> plt.Figure:
    _style()
    if robustness.empty:
        return _empty_saved(
            "Robustness checks",
            "Robustness results are not available.",
            "07_robustness_matrix",
            "Robustness results were not available.",
            output_dir,
        )
    data = robustness.copy()
    if "conclusion_holds" not in data:
        data["conclusion_holds"] = data.select_dtypes(include="bool").any(axis=1)
    table = data.pivot_table(
        index="outcome", columns="specification", values="conclusion_holds", aggfunc="first"
    )
    values = table.astype(float)
    fig, axis = plt.subplots(figsize=(10, max(4, len(table) * 0.4)))
    sns.heatmap(
        values,
        annot=table.map(lambda value: "✓" if value else "✗"),
        fmt="",
        cmap=["#D55E00", "#009E73"],
        cbar=False,
        ax=axis,
    )
    axis.set_title("Robustness conclusions across analysis specifications")
    return _finish(
        fig,
        "07_robustness_matrix",
        "Tick marks indicate whether the stated conclusion holds under each robustness "
        "specification.",
        output_dir,
    )


def confusion_matrix_figure(confusion: pd.DataFrame | np.ndarray, output_dir=None) -> plt.Figure:
    _style()
    fig, axis = plt.subplots(figsize=(6, 5))
    values = confusion.to_numpy() if isinstance(confusion, pd.DataFrame) else np.asarray(confusion)
    sns.heatmap(values, annot=True, fmt="g", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Gold label")
    axis.set_title("Argument classifier confusion matrix")
    return _finish(
        fig,
        "08_classifier_confusion_matrix",
        "Confusion matrix compares gold and predicted argument labels.",
        output_dir,
    )


def generate_all(
    exchanges: pd.DataFrame,
    sentences: pd.DataFrame,
    features: pd.DataFrame,
    tests: pd.DataFrame | None = None,
    robustness: pd.DataFrame | None = None,
    confusion: pd.DataFrame | None = None,
    output_dir: str | Path = "results/figures",
) -> list[plt.Figure]:
    tests = pd.DataFrame() if tests is None else tests
    robustness = pd.DataFrame() if robustness is None else robustness
    confusion = (
        pd.DataFrame(np.zeros((5, 5), dtype=int), index=LABELS, columns=LABELS)
        if confusion is None
        else confusion
    )
    return [
        coverage_figure(exchanges, output_dir),
        argument_mix_figure(sentences, exchanges, output_dir),
        aqs_distribution_figure(features, output_dir),
        forest_figure(tests, output_dir),
        answer_quality_heatmap(features, output_dir),
        answer_ratio_order_figure(features, output_dir),
        robustness_figure(robustness, output_dir),
        confusion_matrix_figure(confusion, output_dir),
    ]
