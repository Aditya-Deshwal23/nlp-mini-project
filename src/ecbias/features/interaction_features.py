"""Measures describing where an exchange occurs within a call."""

from __future__ import annotations

import pandas as pd


def build_interaction_features(exchanges: pd.DataFrame) -> pd.DataFrame:
    """Return order, follow-up, and calendar features for each exchange."""
    result = exchanges[
        ["exchange_id", "call_id", "order_in_call", "order_norm", "is_followup"]
    ].copy()
    analyst = exchanges["analyst_name"].fillna("").astype(str).to_numpy()
    result["n_followups_by_analyst_in_call"] = (
        pd.Series(
            result["is_followup"].to_numpy(),
            index=pd.MultiIndex.from_arrays([result["call_id"], analyst]),
        )
        .groupby(level=[0, 1])
        .transform("sum")
        .to_numpy()
    )
    result["call_n_exchanges"] = result.groupby("call_id")["exchange_id"].transform("size")
    call_dates = exchanges["call_date"] if "call_date" in exchanges else pd.Series(
        pd.NaT, index=exchanges.index
    )
    dates = pd.to_datetime(call_dates, errors="coerce")
    result["call_year"] = dates.dt.year.astype("Int64")
    result["call_quarter"] = dates.dt.quarter.astype("Int64")
    return result
