"""Dataset provenance: content hashes and source lineage for auditability."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ProvenanceRecord:
    dataset_id: str
    content_sha256: str
    sources: list[str]
    created_utc: str
    pipeline_version: str = "trust-aware-insider-v1"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_directory_csvs(data_dir: Path) -> tuple[str, list[str]]:
    """Deterministic hash over sorted CSV file contents under data_dir."""
    paths = sorted(data_dir.rglob("*.csv"))
    h = hashlib.sha256()
    sources: list[str] = []
    for p in paths:
        try:
            blob = p.read_bytes()
        except OSError:
            continue
        h.update(p.name.encode())
        h.update(blob)
        sources.append(str(p.resolve()))
    return h.hexdigest(), sources


def build_provenance_batch(
    data_dir: Path | None,
    synthetic_seed: int | None,
    force_synthetic: bool,
) -> ProvenanceRecord:
    if force_synthetic or data_dir is None or not data_dir.is_dir():
        payload = json.dumps({"mode": "synthetic", "seed": synthetic_seed}, sort_keys=True)
        digest = _hash_bytes(payload.encode())
        return ProvenanceRecord(
            dataset_id=f"synthetic:{digest[:16]}",
            content_sha256=digest,
            sources=["synthetic_cert_like"],
            created_utc=datetime.now(timezone.utc).isoformat(),
            metadata={"seed": synthetic_seed},
        )
    digest, sources = hash_directory_csvs(data_dir)
    return ProvenanceRecord(
        dataset_id=f"cert_batch:{digest[:16]}",
        content_sha256=digest,
        sources=sources[:200],
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata={"root": str(data_dir.resolve())},
    )
