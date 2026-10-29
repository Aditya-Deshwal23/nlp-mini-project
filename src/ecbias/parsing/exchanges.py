"""Turn-to-exchange construction."""

from __future__ import annotations

import re

import pandas as pd


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def build_exchanges(turns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for call_id, group in turns[turns["section"].eq("qa")].groupby("call_id", sort=False):
        group = group.sort_values("turn_idx")
        current: dict | None = None
        seen_analysts: set[str] = set()
        order = 0
        for turn in group.to_dict("records"):
            speaker_type = turn.get("speaker_type", "executive")
            if speaker_type == "operator":
                continue
            if speaker_type == "analyst":
                if current is not None and current["answer_turns"]:
                    rows.append(_finish(current, call_id, order))
                    order += 1
                if current is not None and not current["answer_turns"]:
                    current["question_parts"].append(turn["text"])
                    current["question_turns"] += 1
                    current["was_interrupted"] |= bool(turn.get("was_interrupted", False))
                    continue
                if current is not None and current["analyst_name"] == turn["speaker_name"]:
                    seen_analysts.add(turn["speaker_name"])
                current = {
                    "analyst_name": turn["speaker_name"],
                    "analyst_affiliation": turn.get("speaker_affiliation", ""),
                    "call_date": turn.get("call_date", ""),
                    "question_parts": [turn["text"]],
                    "question_turns": 1,
                    "answer_turns": [],
                    "responders": [],
                    "was_interrupted": bool(turn.get("was_interrupted", False)),
                    "is_followup": turn["speaker_name"] in seen_analysts,
                }
                seen_analysts.add(turn["speaker_name"])
            elif current is not None:
                current["answer_turns"].append(
                    {"speaker": turn["speaker_name"], "text": turn["text"]}
                )
                current["responders"].append(turn["speaker_name"])
                current["was_interrupted"] |= bool(turn.get("was_interrupted", False))
        if current is not None:
            rows.append(_finish(current, call_id, order))
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "exchange_id",
                "call_id",
                "order_in_call",
                "order_norm",
                "analyst_name",
                "analyst_affiliation",
                "question_text",
                "answer_text",
                "answer_turns",
                "responders",
                "n_question_turns",
                "is_followup",
                "was_interrupted",
            ]
        )
    result["exchange_id"] = range(len(result))
    result["order_norm"] = result.groupby("call_id")["order_in_call"].transform(
        lambda values: values / max(1, values.max())
    )
    return result


def _finish(current: dict, call_id: str, order: int) -> dict:
    answer_turns = current["answer_turns"]
    return {
        "exchange_id": -1,
        "call_id": call_id,
        "call_date": current["call_date"],
        "order_in_call": order,
        "order_norm": 0.0,
        "analyst_name": current["analyst_name"],
        "analyst_affiliation": current["analyst_affiliation"],
        "question_text": " ".join(current["question_parts"]),
        "answer_text": " ".join(turn["text"] for turn in answer_turns),
        "answer_turns": answer_turns,
        "responders": list(dict.fromkeys(current["responders"])),
        "n_question_turns": current["question_turns"],
        "is_followup": current["is_followup"],
        "was_interrupted": current["was_interrupted"],
    }
