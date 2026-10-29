"""Classical sentence classifier and honest gold-set evaluation."""

from __future__ import annotations

import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

LABELS = ["CLAIM", "EVIDENCE", "REBUTTAL", "REQUEST", "OTHER"]
_MODALS = {"can", "could", "may", "might", "will", "would", "should", "must"}
_PRONOUNS = {"i", "we", "me", "us", "my", "our"}
_NEGATIONS = {"not", "no", "never", "without"}
_HEDGES = {"may", "might", "could", "possibly", "approximately", "generally", "roughly"}


def hand_features(texts: list[str]) -> np.ndarray:
    values = []
    for text in texts:
        tokens = text.lower().split()
        count = max(1, len(tokens))
        values.append(
            [
                len(tokens),
                "?" in text,
                sum(any(char.isdigit() for char in token) for token in tokens),
                sum(token in _MODALS for token in tokens),
                sum(token in _PRONOUNS for token in tokens),
                sum(token in _NEGATIONS for token in tokens),
                sum(token in _HEDGES for token in tokens),
                sum(token.endswith(("ing", "ed")) for token in tokens) / count,
                sum(token in {"is", "are", "was", "were"} for token in tokens) / count,
            ]
        )
    return np.asarray(values, dtype=float)


def _features(train: list[str], test: list[str] | None = None, char: bool = True):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1 if len(train) < 10 else 3)
    blocks = [("word", word)]
    if char:
        blocks.append(("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)))
    vectorizer = FeatureUnion(blocks)
    try:
        x_train = vectorizer.fit_transform(train)
    except ValueError as error:
        if "no terms remain" not in str(error):
            raise
        vectorizer = FeatureUnion(
            [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1))]
            + (
                [("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1))]
                if char
                else []
            )
        )
        x_train = vectorizer.fit_transform(train)
    x_test = vectorizer.transform(test) if test is not None else None
    return (
        hstack([x_train, hand_features(train)]),
        (hstack([x_test, hand_features(test)]) if test is not None else None),
        vectorizer,
    )


def train_and_evaluate(
    gold: pd.DataFrame,
    rules: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 42,
    regime: str = "gold_cv",
    model_name: str = "logistic",
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gold = gold[gold["gold_label"].isin(LABELS)].copy()
    if gold.empty:
        raise ValueError("No completed gold_label rows available")
    test_texts = gold["text"].tolist()
    y = gold["gold_label"].tolist()
    fit_frame = rules if regime == "silver" else gold
    if "gold_label" not in fit_frame:
        fit_frame = fit_frame.assign(gold_label=fit_frame["label_rule"])
    texts = fit_frame["text"].tolist()
    train_y = fit_frame["gold_label"].tolist()
    x, x_test, vectorizer = _features(texts, test_texts)
    model = (
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        if model_name == "logistic"
        else LinearSVC(class_weight="balanced", random_state=seed)
    )
    if regime == "gold_cv" and min(pd.Series(y).value_counts()) >= 2:
        folds = min(5, int(pd.Series(y).value_counts().min()))
        predictions = cross_val_predict(
            model, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed)
        )
    else:
        model.fit(x, train_y)
        predictions = model.predict(x_test if regime == "silver" else x)
    model.fit(x, train_y)
    rule_predictions = (
        rules["label_rule"].tolist()
        if "label_rule" in rules.columns and len(rules) == len(gold)
        else (
            gold["label_rule"].tolist() if "label_rule" in gold.columns else ["OTHER"] * len(gold)
        )
    )
    hybrid_predictions = [
        ml if rule == "OTHER" else rule for rule, ml in zip(rule_predictions, predictions)
    ]
    eval_rows = []
    for method, method_predictions in (
        ("rules", rule_predictions),
        (model_name, predictions),
        ("hybrid", hybrid_predictions),
    ):
        method_report = classification_report(
            y, method_predictions, labels=LABELS, output_dict=True, zero_division=0
        )
        method_matrix = confusion_matrix(y, method_predictions, labels=LABELS)
        matrix_rows = []
        for actual, row in zip(LABELS, method_matrix):
            for predicted, count in zip(LABELS, row):
                matrix_rows.append(
                    {
                        "method": method,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(count),
                    }
                )
        matrix_path = output / "confusion_matrix.csv"
        matrix_frame = pd.DataFrame(matrix_rows)
        if matrix_path.exists():
            matrix_frame = pd.concat([pd.read_csv(matrix_path), matrix_frame], ignore_index=True)
        matrix_frame.to_csv(matrix_path, index=False)
        for label in ["macro", *LABELS]:
            key = "macro avg" if label == "macro" else label
            eval_rows.append(
                {
                    "method": method,
                    "regime": regime,
                    "label": label,
                    "precision": method_report[key]["precision"],
                    "recall": method_report[key]["recall"],
                    "f1": method_report[key]["f1-score"],
                }
            )
    pd.DataFrame(eval_rows).to_csv(output / "argmining_eval.csv", index=False)
    errors = gold.assign(predicted=predictions)
    errors = errors[errors["gold_label"] != errors["predicted"]].head(30)
    flags = [column for column in errors if column.startswith("has_") or column == "is_question"]
    errors[["text", "gold_label", "predicted", *flags]].to_csv(
        output / "error_analysis.csv", index=False
    )
    feature_names = vectorizer.get_feature_names_out().tolist() + [
        "token_count",
        "question_mark",
        "numeric_count",
        "modal_count",
        "first_person_count",
        "negation_count",
        "hedge_count",
        "verb_ing_ed_prop",
        "copula_prop",
    ]
    feature_rows = []
    for index, label in enumerate(model.classes_):
        order = np.argsort(model.coef_[index])[-20:][::-1]
        feature_rows.extend(
            {
                "label": label,
                "feature": feature_names[position],
                "weight": float(model.coef_[index, position]),
            }
            for position in order
        )
    pd.DataFrame(feature_rows).to_csv(output / "top_features_by_class.csv", index=False)
    models_dir = output.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "vectorizer": vectorizer}, models_dir / "argmining_model.joblib")
    metadata = {
        "library_versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "seed": seed,
        "date": date.today().isoformat(),
        "training_regime": regime,
        "model": model_name,
    }
    (models_dir / "argmining_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "macro_f1": float(f1_score(y, predictions, labels=LABELS, average="macro")),
        "predictions": predictions,
        "matrix": confusion_matrix(y, predictions, labels=LABELS),
    }


def hybrid_label(rule_label: str, ml_label: str) -> str:
    return ml_label if rule_label == "OTHER" else rule_label
