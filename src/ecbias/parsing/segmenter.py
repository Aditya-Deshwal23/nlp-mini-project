"""Q&A section segmentation."""

from __future__ import annotations

import logging
import re

import pandas as pd

LOGGER = logging.getLogger(__name__)


def segment_qa(df: pd.DataFrame, markers: list[str]) -> tuple[pd.DataFrame, dict[str, bool]]:
    result = df.copy()
    statuses: dict[str, bool] = {}
    for call_id, group in result.groupby("call_id", sort=False):
        pattern = "|".join(re.escape(marker) for marker in markers)
        mask = group["text"].str.contains(pattern, case=False, na=False, regex=True)
        found = bool(mask.any())
        statuses[str(call_id)] = found
        if not found:
            LOGGER.warning("No Q&A marker found for call %s", call_id)
            result.loc[group.index, "section"] = "unknown"
            continue
        start = group.index[mask][0]
        result.loc[group.index[group.index >= start], "section"] = "qa"
    result["qa_found"] = result["call_id"].map(statuses)
    return result, statuses
