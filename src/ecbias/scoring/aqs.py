"""Transparent Argument Quality Score (AQS) and sensitivity scorers.

AQS is an interpretable proxy rubric, not ground truth or an objective measure
of question or answer quality. Components are percentile-ranked over the
current corpus before applying configured weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import percentileofscore, spearmanr
from sklearn.decomposition import PCA


def load_aqs_config(path: str | Path = "configs/aqs.yaml") -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for side in ("question", "answer"):
        weights = config[side]["weights"]
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(f"{side}.weights must sum to one")
    return config


def _percentile(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    return series.map(
        lambda value: (
            np.nan if pd.isna(value) else percentileofscore(valid, value, kind="rank") / 100
        )
    )


def _component_frame(features: pd.DataFrame, config: dict, side: str) -> pd.DataFrame:
    band_low, band_high = config[side]["readability_band"]
    threshold = config[side]["hedge_threshold"]
    frame = features[features["side"].eq(side)].copy().set_index("exchange_id")
    structure = (
        frame["n_claim"].gt(0).astype(float)
        + frame["n_evidence"].gt(0).astype(float)
        + frame["claim_supported"].astype(float)
    ) / 3
    specificity = (_percentile(frame["numeric_density"]) + _percentile(frame["entity_density"])) / 2
    readability = frame["flesch_reading_ease"].map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else max(
                0,
                1 - min(abs(value - (band_low + band_high) / 2) / ((band_high - band_low) / 2), 1),
            )
        )
    )
    clarity = (
        readability
        + frame["long_sentence_count"].map(
            lambda value: max(0, 1 - value / max(frame["n_sentences"].max(), 1))
        )
    ) / 2
    calibration = frame["hedge_rate"].map(
        lambda value: (
            np.nan
            if pd.isna(value)
            else max(0, 1 - max(0, value - threshold) / max(threshold, 1e-9))
        )
    )
    result = pd.DataFrame(
        {
            "structure": structure,
            "specificity": specificity,
            "clarity": clarity,
            "calibration": calibration,
        },
        index=frame.index,
    )
    if side == "question":
        result["focus"] = frame["focus"]
    weights = config[side]["weights"]
    result[f"AQS_{'q' if side == 'question' else 'a'}"] = (
        sum(result[name] * weight for name, weight in weights.items()) * 100
    )
    return result.reset_index()


def score_aqs(features: pd.DataFrame, config_path: str | Path = "configs/aqs.yaml") -> pd.DataFrame:
    config = load_aqs_config(config_path)
    questions = _component_frame(features, config, "question")
    answers = _component_frame(features, config, "answer")
    return questions.merge(answers, on="exchange_id", how="outer", suffixes=("_q", "_a"))


def alternative_scorers(components: pd.DataFrame, side: str) -> dict[str, pd.Series]:
    names = [
        name
        for name in ("structure", "specificity", "clarity", "calibration", "focus")
        if name in components
    ]
    matrix = components[names].fillna(components[names].median()).to_numpy()
    equal = pd.Series(matrix.mean(axis=1), index=components.index, name="equal")
    if len(matrix) < 2 or np.all(np.nanstd(matrix, axis=0) == 0):
        pca_values = matrix.mean(axis=1)
    else:
        pca_values = PCA(n_components=1, random_state=42).fit_transform(matrix).ravel()
    if len(pca_values) > 1 and np.std(pca_values) and np.std(matrix.mean(axis=1)):
        if np.corrcoef(pca_values, matrix.mean(axis=1))[0, 1] < 0:
            pca_values *= -1
    pca = pd.Series(pca_values, index=components.index, name="pca")
    ranks = pd.DataFrame(matrix, columns=names).rank(pct=True).mean(axis=1)
    mean_rank = pd.Series(ranks.to_numpy(), index=components.index, name="mean_rank")
    return {"equal": equal, "pca": pca, "mean_rank": mean_rank}


def scorer_agreement(scores: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for left_name, left in scores.items():
        for right_name, right in scores.items():
            if left_name < right_name:
                comparable = pd.concat([left, right], axis=1).dropna()
                if len(comparable) < 2:
                    rho = np.nan
                elif comparable.iloc[:, 0].nunique() < 2 or comparable.iloc[:, 1].nunique() < 2:
                    rho = 1.0 if comparable.iloc[:, 0].equals(comparable.iloc[:, 1]) else 0.0
                else:
                    rho = spearmanr(comparable.iloc[:, 0], comparable.iloc[:, 1]).statistic
                rows.append({"scorer_a": left_name, "scorer_b": right_name, "spearman_rho": rho})
    return pd.DataFrame(rows)
