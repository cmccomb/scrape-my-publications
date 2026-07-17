"""Tests for the split local-collector and trusted-uploader pipeline."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import datasets
import pandas
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import write_publication_snapshot
from upload_snapshot import load_and_validate_snapshot


def _publication_dataset() -> datasets.Dataset:
    frame = pandas.DataFrame(
        [
            {
                "author_pub_id": "author:one",
                "bib_dict": {"title": "One", "author": "A. Author"},
                "embedding": [0.1, 0.2, 0.3],
                "num_citations": 4,
                "Last Updated": "2026-07-17",
            },
            {
                "author_pub_id": "author:two",
                "bib_dict": {"title": "Two", "author": "B. Author"},
                "embedding": [0.4, 0.5, 0.6],
                "num_citations": 2,
                "Last Updated": "2026-07-17",
            },
        ]
    )
    return datasets.Dataset.from_pandas(frame, preserve_index=False)


def _write_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_path = tmp_path / "publications.parquet"
    write_publication_snapshot(
        _publication_dataset(),
        snapshot_path,
        discovered_publications=2,
        new_publications=1,
        stale_publications=0,
        metadata_only_publications=1,
        refreshed_publications=1,
        refresh_mode="full",
    )
    return snapshot_path, snapshot_path.with_suffix(".parquet.manifest.json")


def test_snapshot_round_trip_is_valid(tmp_path: Path) -> None:
    """A fresh collector snapshot should pass uploader validation."""

    snapshot_path, manifest_path = _write_snapshot(tmp_path)
    frame, manifest = load_and_validate_snapshot(snapshot_path, manifest_path)

    assert frame["author_pub_id"].tolist() == ["author:one", "author:two"]
    assert manifest["dataset_rows"] == 2
    assert manifest["snapshot_sha256"] == hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()


def test_snapshot_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    """The uploader must reject a snapshot changed after collection."""

    snapshot_path, manifest_path = _write_snapshot(tmp_path)
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b"tampered")

    with pytest.raises(RuntimeError, match="checksum"):
        load_and_validate_snapshot(snapshot_path, manifest_path)


def test_stale_snapshot_is_rejected(tmp_path: Path) -> None:
    """Old snapshots cannot be replayed through the protected uploader."""

    snapshot_path, manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at_utc"] = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=8)
    ).isoformat()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="too old"):
        load_and_validate_snapshot(snapshot_path, manifest_path)


def test_full_snapshot_must_match_discovered_profile(tmp_path: Path) -> None:
    """A full baseline cannot silently omit a discovered publication."""

    snapshot_path, manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["discovered_publications"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="exactly mirror"):
        load_and_validate_snapshot(snapshot_path, manifest_path)
