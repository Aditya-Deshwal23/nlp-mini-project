"""Privacy-safe exchange enrichment and coverage outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .gender import infer_gender, load_overrides, speaker_id


def enrich_exchanges(
    exchanges: pd.DataFrame,
    turns: pd.DataFrame,
    salt: str,
    overrides_path: str | Path,
) -> pd.DataFrame:
    overrides = load_overrides(overrides_path)
    result = exchanges.copy()
    turn_lookup = turns.set_index(["call_id", "speaker_name"], drop=False)

    def person(name: str) -> tuple[str, str, str]:
        gender, confidence = infer_gender(name, overrides)
        return gender, confidence, speaker_id(name, salt)

    analyst = result["analyst_name"].map(person)
    result["analyst_gender"] = analyst.map(lambda item: item[0])
    result["analyst_gender_conf"] = analyst.map(lambda item: item[1])
    result["analyst_id"] = analyst.map(lambda item: item[2])
    first_ids, first_roles, first_genders, mixes, counts = [], [], [], [], []
    for row in result.itertuples():
        responders = list(row.responders)
        metadata = []
        for name in responders:
            gender, confidence, identity = person(name)
            role = ""
            key = (row.call_id, name)
            if key in turn_lookup.index:
                found = turn_lookup.loc[key]
                if isinstance(found, pd.DataFrame):
                    found = found.iloc[0]
                role = str(found.get("executive_role", ""))
            metadata.append((identity, role, gender, confidence))
        first = metadata[0] if metadata else ("", "", "Unknown", "unknown")
        genders = {item[2] for item in metadata}
        mix = (
            "all_F"
            if genders == {"F"}
            else "all_M"
            if genders == {"M"}
            else "mixed"
            if genders - {"Unknown"}
            else "unknown"
        )
        first_ids.append(first[0])
        first_roles.append(first[1])
        first_genders.append(first[2])
        mixes.append(mix)
        counts.append(len(responders))
    result["first_responder_id"] = first_ids
    result["first_responder_role"] = first_roles
    result["first_responder_gender"] = first_genders
    result["responder_gender_mix"] = mixes
    result["n_responders"] = counts
    return result


def write_gender_coverage(exchanges: pd.DataFrame, output: str | Path) -> None:
    rows = []
    for population, column in (
        ("analyst", "analyst_gender"),
        ("executive", "first_responder_gender"),
    ):
        counts = exchanges[column].value_counts()
        total = max(1, len(exchanges))
        for gender in ("F", "M", "Unknown"):
            rows.append(
                {
                    "kind": "population",
                    "population": population,
                    "gender": gender,
                    "count": int(counts.get(gender, 0)),
                    "share": counts.get(gender, 0) / total,
                }
            )
    cells = exchanges.groupby(["analyst_gender", "first_responder_gender"]).size()
    for (analyst_gender, executive_gender), count in cells.items():
        rows.append(
            {
                "kind": "cell",
                "population": "",
                "gender": "",
                "analyst_gender": analyst_gender,
                "executive_gender": executive_gender,
                "count": int(count),
                "share": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(output, index=False)
