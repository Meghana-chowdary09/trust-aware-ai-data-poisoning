"""
Spark / Flink hook (stub). In production, replace with:

  spark.read.csv(...).groupBy("user_id").agg(...)

`ingestion_layer.ingest_batch_files` already records `spark_equivalent` for observability.
This module exists so the architecture diagram’s “Spark” stage has a named extension point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_events_pandas_stand_in(csv_glob: str) -> pd.DataFrame:
    """Local single-node aggregate; swap for Spark DataFrame operations in cluster."""
    paths = sorted(Path().glob(csv_glob))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths])
