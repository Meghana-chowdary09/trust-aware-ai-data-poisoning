"""
Ingestion layer: batch files (local / uploaded) and optional Kafka-style micro-batches.

Production stacks often use Kafka + Spark; here we provide:
  - BatchFileIngestion: scan CSVs and emit logical micro-batches (pandas stand-in for Spark).
  - Optional Kafka consumer when kafka-python is installed and env KAFKA_TOPIC is set.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd


@dataclass
class IngestionManifest:
    transport: str
    micro_batches: int
    total_csv_rows: int
    files_seen: int
    spark_equivalent: str

    def to_dict(self) -> dict:
        return asdict(self)


def _count_csv_rows(root: Path) -> tuple[int, int]:
    rows = 0
    nfiles = 0
    for p in root.rglob("*.csv"):
        try:
            # Header + count lines without loading full file into memory
            with p.open("rb") as f:
                rows += sum(1 for _ in f) - 1
            nfiles += 1
        except OSError:
            continue
    return max(0, rows), nfiles


def ingest_batch_files(
    data_dir: Path | None,
    micro_batch_rows: int = 50_000,
    audit_hook: Callable[[str, dict], None] | None = None,
) -> IngestionManifest:
    """
    Simulate Spark structured streaming: walk CSVs, count rows, emit batch metadata.
    When no directory, synthetic path reports zero raw rows (features built in-memory later).
    """
    if audit_hook:
        audit_hook("ingestion", {"event": "batch_scan_start", "path": str(data_dir)})

    if data_dir is None or not data_dir.is_dir():
        m = IngestionManifest(
            transport="in_memory_synthetic",
            micro_batches=1,
            total_csv_rows=0,
            files_seen=0,
            spark_equivalent="pandas_feature_generation",
        )
        if audit_hook:
            audit_hook("ingestion", {"event": "batch_scan_done", **m.to_dict()})
        return m

    total_rows, nfiles = _count_csv_rows(data_dir)
    batches = max(1, (total_rows + micro_batch_rows - 1) // micro_batch_rows)
    m = IngestionManifest(
        transport="batch_csv_files",
        micro_batches=int(batches),
        total_csv_rows=int(total_rows),
        files_seen=int(nfiles),
        spark_equivalent="pandas_aggregate_per_user",
    )
    if audit_hook:
        audit_hook("ingestion", {"event": "batch_scan_done", **m.to_dict()})
    return m


def kafka_microbatch_iterator(
    topic: str | None,
    max_messages: int = 100,
) -> Iterator[dict]:
    """
    Optional Kafka poll. If kafka-python missing or KAFKA_BOOTSTRAP_SERVERS unset, yields nothing.
    """
    topic = topic or os.environ.get("KAFKA_TOPIC")
    brokers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if not topic or not brokers:
        return
    try:
        from kafka import KafkaConsumer  # type: ignore
    except ImportError:
        return

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=brokers.split(","),
        consumer_timeout_ms=3000,
        auto_offset_reset="latest",
    )
    n = 0
    for msg in consumer:
        yield {"offset": msg.offset, "partition": msg.partition, "value": msg.value}
        n += 1
        if n >= max_messages:
            break
    consumer.close()


def spark_compatible_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Single-node substitute for Spark DataFrame read; swap for spark.read.csv in cluster."""
    return pd.read_csv(path, low_memory=False, **kwargs)
