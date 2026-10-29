"""Resumable terminal annotation and annotator agreement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = {"1": "CLAIM", "2": "EVIDENCE", "3": "REBUTTAL", "4": "REQUEST", "5": "OTHER"}


class TerminalAnnotator:
    def __init__(self, path: str | Path, annotator: str):
        self.path = Path(path)
        self.annotator = annotator
        if self.path.exists():
            self.frame = pd.read_csv(self.path)
        else:
            raise FileNotFoundError(self.path)
        column = (
            "gold_label" if annotator in {"", "annotator1", "a1"} else f"gold_label_{annotator}"
        )
        if column not in self.frame:
            self.frame[column] = ""
        self.column = column

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(self.path, index=False)

    def run(self, input_fn=input, output_fn=print) -> None:
        index = 0
        history: list[int] = []
        while index < len(self.frame):
            if str(self.frame.at[index, self.column]).strip():
                index += 1
                continue
            start = max(0, index - 1)
            end = min(len(self.frame), index + 2)
            context = self.frame.iloc[start:end]["text"].tolist()
            output_fn(f"\n[{index + 1}/{len(self.frame)}] " + " | ".join(context))
            key = (
                input_fn("1 CLAIM  2 EVIDENCE  3 REBUTTAL  4 REQUEST  5 OTHER  s skip  u undo: ")
                .strip()
                .lower()
            )
            if key == "u":
                if history:
                    index = history.pop()
                    self.frame.at[index, self.column] = ""
                    self.save()
                continue
            if key == "s":
                history.append(index)
                self.frame.at[index, self.column] = "SKIP"
                self.save()
                index += 1
                continue
            if key not in LABELS:
                output_fn("Please enter 1-5, s, or u.")
                continue
            history.append(index)
            self.frame.at[index, self.column] = LABELS[key]
            self.save()
            index += 1


def agreement(path: str | Path, first: str, second: str) -> float:
    frame = pd.read_csv(path)
    left, right = f"gold_label_{first}", f"gold_label_{second}"
    if left not in frame or right not in frame:
        raise ValueError(f"Both annotator columns are required: {left}, {right}")
    comparable = frame[[left, right]].replace("", pd.NA).dropna()
    if comparable.empty:
        raise ValueError("No overlapping annotations")
    return float(cohen_kappa_score(comparable[left], comparable[right]))


def annotator_path(base: str | Path, annotator: str) -> Path:
    """Return the base file for the first annotator and a separate file thereafter."""
    base = Path(base)
    if annotator in {"", "annotator1", "a1"}:
        return base
    return base.with_name(f"{base.stem}_{annotator}{base.suffix}")


def agreement_files(base: str | Path, first: str, second: str) -> float:
    left = pd.read_csv(annotator_path(base, first))
    right = pd.read_csv(annotator_path(base, second))
    column = "gold_label" if "gold_label" in left else f"gold_label_{first}"
    right_column = "gold_label" if "gold_label" in right else f"gold_label_{second}"
    comparable = pd.DataFrame({"first": left[column], "second": right[right_column]})
    comparable = comparable.replace({"": pd.NA, "SKIP": pd.NA}).dropna()
    if comparable.empty:
        raise ValueError("No overlapping annotations")
    return float(cohen_kappa_score(comparable["first"], comparable["second"]))
