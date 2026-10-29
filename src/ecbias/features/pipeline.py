"""Assembly of the privacy-safe exchange feature table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from .interaction_features import build_interaction_features
from .response_features import build_response_features

LOGGER = logging.getLogger(__name__)


def build_feature_table(
    exchanges: pd.DataFrame,
    quality: pd.DataFrame | None = None,
    spec_path: str | Path = "configs/feature_spec.yaml",
    output: str | Path | None = None,
) -> pd.DataFrame:
    """Merge configured enrichment, response, interaction, and quality columns."""
    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    response = build_response_features(exchanges, spec_path)
    interaction = build_interaction_features(exchanges)
    result = exchanges.merge(response, on="exchange_id").merge(
        interaction, on=["exchange_id", "call_id", "order_in_call", "order_norm", "is_followup"]
    )
    if quality is not None:
        result = result.merge(
            quality.drop(columns=["analyst_id"], errors="ignore"),
            on="exchange_id",
            how="left",
        )
    columns = [column["name"] for column in spec["columns"]]
    missing = sorted(set(columns) - set(result.columns))
    if missing:
        raise ValueError(f"Feature table is missing configured columns: {', '.join(missing)}")
    result = result[columns]
    nullable = set(spec["missing_data_policy"]["nullable_columns"])
    observed_missing = sorted(set(result.columns[result.isna().any()]) - nullable)
    if observed_missing:
        LOGGER.warning("Unexpected missing values in non-nullable columns: %s", observed_missing)
    LOGGER.info(
        "Feature missing values: %s", result.isna().sum().loc[lambda values: values.gt(0)].to_dict()
    )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False)
    return result
