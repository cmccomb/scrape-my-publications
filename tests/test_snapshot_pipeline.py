"""Tests for the split local-collector and trusted-uploader pipeline."""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pandas
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import write_publication_snapshot
import upload_snapshot as snapshot_uploader
from upload_snapshot import (
    build_dataset_metadata,
    load_and_validate_snapshot,
    prepare_publication_dataset,
)


def _publication_dataset() -> datasets.Dataset:
    frame = pandas.DataFrame(
        [
            {
                "author_pub_id": "author:one",
                "bib_dict": {"title": "One", "author": "A. Author"},
                "embedding": [0.1, 0.2, 0.3],
                "num_citations": 4,
                "Last Updated": "2026-07-17",
                "2026": 4.0,
            },
            {
                "author_pub_id": "author:two",
                "bib_dict": {"title": "Two", "author": "B. Author"},
                "embedding": [0.4, 0.5, 0.6],
                "num_citations": 2,
                "Last Updated": "2026-07-17",
                "2026": 2.0,
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


def test_hub_dataset_uses_canonical_embedding_type() -> None:
    """The uploaded Parquet schema should match the declared Hub schema."""

    publication_dataset = prepare_publication_dataset(
        _publication_dataset().to_pandas()
    )

    assert publication_dataset.features["embedding"] == datasets.List(
        datasets.Value("float32")
    )


def test_hub_metadata_replaces_features_and_split_details() -> None:
    """Current year columns and row counts must be published to the card."""

    publication_dataset = prepare_publication_dataset(
        _publication_dataset().to_pandas()
    )
    metadata = build_dataset_metadata(publication_dataset)["dataset_info"]
    feature_names = [feature["name"] for feature in metadata["features"]]

    assert "2026" in feature_names
    assert metadata["splits"] == [{
        "name": "train",
        "num_bytes": publication_dataset.data.nbytes,
        "num_examples": 2,
    }]


def test_hub_metadata_preserves_calculated_size_values() -> None:
    """A feature refresh must not erase Hub's calculated size metadata."""

    publication_dataset = prepare_publication_dataset(
        _publication_dataset().to_pandas()
    )
    hub_info = publication_dataset.info.copy()
    hub_info.download_size = 100
    hub_info.dataset_size = 200
    hub_info.size_in_bytes = 300
    metadata = build_dataset_metadata(
        publication_dataset,
        dataset_info=hub_info,
    )["dataset_info"]

    assert metadata["download_size"] == 100
    assert metadata["dataset_size"] == 200


def test_upload_replaces_stale_hub_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The trusted upload must publish schema metadata after the data commit."""

    snapshot_path, manifest_path = _write_snapshot(tmp_path)
    captured: dict[str, object] = {}

    def fake_push_to_hub(
        publication_dataset: datasets.Dataset,
        repo_id: str,
        **kwargs: object,
    ) -> SimpleNamespace:
        captured["features"] = publication_dataset.features
        captured["repo_id"] = repo_id
        return SimpleNamespace(oid="data-commit")

    def fake_metadata_update(
        repo_id: str,
        metadata: dict[str, object],
        **kwargs: object,
    ) -> str:
        captured["metadata_repo_id"] = repo_id
        captured["metadata"] = metadata
        captured["parent_commit"] = kwargs["parent_commit"]
        return "https://huggingface.co/datasets/ccm/publications/commit/final-commit"

    stale_info = prepare_publication_dataset(
        _publication_dataset().to_pandas()
    ).info.copy()
    stale_features = stale_info.features.copy()
    stale_features.pop("2026")
    stale_info.features = stale_features
    stale_info.download_size = 100
    stale_info.dataset_size = 200
    stale_info.size_in_bytes = 300

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            captured["token"] = token

        def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            captured["revision"] = revision
            return SimpleNamespace(sha="final-commit")

    monkeypatch.setattr(datasets.Dataset, "push_to_hub", fake_push_to_hub)
    monkeypatch.setattr(snapshot_uploader, "metadata_update", fake_metadata_update)
    monkeypatch.setattr(snapshot_uploader, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        snapshot_uploader,
        "repair_hub_dataset_metadata_if_needed",
        lambda *, hf_token: stale_info,
    )
    monkeypatch.setattr(
        snapshot_uploader,
        "get_hub_dataset_info",
        lambda *, hf_token: stale_info,
    )

    commit, _manifest = snapshot_uploader.upload_snapshot(
        snapshot_path,
        manifest_path,
        hf_token="test-token",
    )

    assert commit == "final-commit"
    assert captured["repo_id"] == "ccm/publications"
    assert captured["metadata_repo_id"] == "ccm/publications"
    assert captured["parent_commit"] == "data-commit"
    assert captured["revision"] == "main"
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["dataset_info"]["download_size"] == 100
    assert metadata["dataset_info"]["dataset_size"] == 200


def test_legacy_missing_sizes_are_repaired_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy card with null sizes is repaired using the Hub's live data."""

    existing_dataset = prepare_publication_dataset(
        _publication_dataset().to_pandas()
    )
    stale_info = existing_dataset.info.copy()
    stale_info.download_size = None
    stale_info.dataset_size = None
    stale_info.size_in_bytes = None
    captured: dict[str, object] = {}

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            captured["token"] = token

        def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
            assert repo_id == "ccm/publications"
            assert repo_type == "dataset"
            return True

        def dataset_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
            assert repo_id == "ccm/publications"
            assert revision == "main"
            return SimpleNamespace(sha="legacy-commit")

        def list_repo_tree(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(path="README.md", size=4),
                SimpleNamespace(path="data/train-00000.parquet", size=101),
                SimpleNamespace(path="data/train-00001.parquet", size=202),
            ]

    def fake_metadata_update(
        repo_id: str,
        metadata: dict[str, object],
        **kwargs: object,
    ) -> str:
        captured["repo_id"] = repo_id
        captured["metadata"] = metadata
        captured["parent_commit"] = kwargs["parent_commit"]
        return "commit-url"

    monkeypatch.setattr(snapshot_uploader, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        snapshot_uploader,
        "get_hub_dataset_info",
        lambda *, hf_token: stale_info,
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: existing_dataset)
    monkeypatch.setattr(snapshot_uploader, "metadata_update", fake_metadata_update)

    repaired_info = snapshot_uploader.repair_hub_dataset_metadata_if_needed(
        hf_token="test-token"
    )

    assert repaired_info is not None
    assert repaired_info.download_size == 303
    assert repaired_info.dataset_size == existing_dataset.data.nbytes
    assert repaired_info.size_in_bytes == 303 + existing_dataset.data.nbytes
    assert captured["repo_id"] == "ccm/publications"
    assert captured["parent_commit"] == "legacy-commit"
