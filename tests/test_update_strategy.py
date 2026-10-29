"""Unit tests for the publication refresh strategy."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as publication_sync
from main import (
    determine_publications_to_refresh,
    ensure_last_updated_column,
    load_specter2_model,
    synchronize_listing_metadata,
    validate_publication_dataset,
)


def test_specter2_loader_uses_pinned_base_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The embedding loader must avoid adapters and remote model code."""

    calls: list[tuple[str, str, dict[str, object]]] = []
    tokenizer = object()
    model = SimpleNamespace(eval=lambda: calls.append(("model", "eval", {})))

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda name, **kwargs: (
                calls.append(("tokenizer", name, kwargs)),
                tokenizer,
            )[1]
        ),
        AutoModel=SimpleNamespace(
            from_pretrained=lambda name, **kwargs: (
                calls.append(("model", name, kwargs)),
                model,
            )[1]
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    loaded_tokenizer, loaded_model = load_specter2_model()

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert calls == [
        (
            "tokenizer",
            publication_sync.SPECTER_MODEL_NAME,
            {
                "revision": publication_sync.SPECTER_MODEL_REVISION,
                "trust_remote_code": False,
            },
        ),
        (
            "model",
            publication_sync.SPECTER_MODEL_NAME,
            {
                "revision": publication_sync.SPECTER_MODEL_REVISION,
                "trust_remote_code": False,
            },
        ),
        ("model", "eval", {}),
    ]


def test_selects_new_publications_when_available() -> None:
    """New Google Scholar publications should take priority."""

    remote_publications = [
        {"author_pub_id": "pub1"},
        {"author_pub_id": "pub2"},
        {"author_pub_id": "pub3"},
    ]
    existing_dataset = pandas.DataFrame(
        [{"author_pub_id": "pub1", "Last Updated": "2024-01-01"}]
    )

    plan = determine_publications_to_refresh(
        remote_publications, existing_dataset, stale_refresh_limit=2
    )

    assert plan.new_publication_ids == ["pub2", "pub3"]
    assert plan.stale_publication_ids == ["pub1"]


def test_refreshes_stalest_publications_when_no_new_entries() -> None:
    """The stalest entries should be refreshed when nothing new exists."""

    remote_publications = [
        {"author_pub_id": "pub1"},
        {"author_pub_id": "pub2"},
        {"author_pub_id": "pub3"},
    ]
    existing_dataset = pandas.DataFrame(
        [
            {"author_pub_id": "pub1", "Last Updated": "2024-01-03"},
            {"author_pub_id": "pub2", "Last Updated": "2023-12-01"},
            {"author_pub_id": "pub3", "Last Updated": "2023-12-01"},
        ]
    )

    plan = determine_publications_to_refresh(
        remote_publications, existing_dataset, stale_refresh_limit=2
    )

    assert plan.new_publication_ids == []
    assert plan.stale_publication_ids == ["pub2", "pub3"]


def test_includes_stale_refreshes_when_new_publications_exist() -> None:
    """New publications should not block background refreshes."""

    remote_publications = [
        {"author_pub_id": "pub1"},
        {"author_pub_id": "pub2"},
        {"author_pub_id": "pub3"},
        {"author_pub_id": "pub4"},
    ]
    existing_dataset = pandas.DataFrame(
        [
            {"author_pub_id": "pub1", "Last Updated": "2024-01-03"},
            {"author_pub_id": "pub2", "Last Updated": "2023-12-01"},
            {"author_pub_id": "pub3", "Last Updated": "2023-12-02"},
        ]
    )

    plan = determine_publications_to_refresh(
        remote_publications, existing_dataset, stale_refresh_limit=1
    )

    assert plan.new_publication_ids == ["pub4"]
    assert plan.stale_publication_ids == ["pub2"]


def test_last_updated_column_added_when_missing() -> None:
    """A missing ``Last Updated`` column should be created."""

    dataset = pandas.DataFrame(
        [
            {"author_pub_id": "pub1"},
            {"author_pub_id": "pub2", "Last Updated": datetime.date(2024, 1, 1)},
        ]
    )

    result = ensure_last_updated_column(dataset)

    assert "Last Updated" in result.columns
    assert result.loc[0, "Last Updated"] == ""
    assert result.loc[1, "Last Updated"] == datetime.date(2024, 1, 1)


def test_new_publication_backlog_is_bounded() -> None:
    """A large backlog should be spread over multiple rate-limited runs."""

    remote_publications = [
        {"author_pub_id": f"pub{index}"}
        for index in range(6)
    ]
    existing_dataset = pandas.DataFrame(columns=["author_pub_id", "Last Updated"])

    plan = determine_publications_to_refresh(
        remote_publications,
        existing_dataset,
        stale_refresh_limit=0,
        new_refresh_limit=3,
    )

    assert plan.new_publication_ids == ["pub0", "pub1", "pub2"]


def test_full_refresh_selects_every_current_publication() -> None:
    """Full refreshes should ignore incremental limits and cover all records."""

    remote_publications = [
        {"author_pub_id": "existing-2"},
        {"author_pub_id": "new-1"},
        {"author_pub_id": "existing-1"},
        {"author_pub_id": "new-2"},
    ]
    existing_dataset = pandas.DataFrame(
        [
            {"author_pub_id": "existing-1", "Last Updated": "2026-01-01"},
            {"author_pub_id": "existing-2", "Last Updated": "2025-01-01"},
            {"author_pub_id": "removed", "Last Updated": "2026-01-01"},
        ]
    )

    plan = determine_publications_to_refresh(
        remote_publications,
        existing_dataset,
        stale_refresh_limit=0,
        new_refresh_limit=1,
        full_refresh=True,
    )

    assert plan.new_publication_ids == ["new-1", "new-2"]
    assert plan.stale_publication_ids == []
    assert plan.metadata_only_publication_ids == ["existing-2", "existing-1"]
    assert plan.ordered_ids == ["new-1", "new-2"]


def test_full_refresh_prunes_and_updates_listing_metadata() -> None:
    """A full baseline should mirror the profile without re-embedding old rows."""

    existing_dataset = pandas.DataFrame(
        [
            {
                "author_pub_id": "current",
                "num_citations": 2,
                "embedding": [0.1, 0.2],
                "Last Updated": "2025-01-01",
            },
            {
                "author_pub_id": "removed",
                "num_citations": 10,
                "embedding": [0.3, 0.4],
                "Last Updated": "2025-01-01",
            },
        ]
    )
    remote_publications = [
        {
            "author_pub_id": "current",
            "num_citations": 7,
            "citedby_url": "https://example.test/citations",
        },
        {"author_pub_id": "new", "num_citations": 1},
    ]

    synchronized = synchronize_listing_metadata(
        existing_dataset,
        remote_publications,
    )

    assert synchronized["author_pub_id"].tolist() == ["current"]
    assert synchronized.loc[0, "num_citations"] == 7
    assert synchronized.loc[0, "embedding"] == [0.1, 0.2]
    assert synchronized.loc[0, "Last Updated"] == "2025-01-01"


def test_retry_operation_uses_bounded_backoff() -> None:
    """Transient Scholar failures should retry without busy-looping."""

    attempts = 0
    delays: list[float] = []

    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary")
        return "ok"

    result = publication_sync.retry_operation(
        flaky_operation,
        description="test operation",
        attempts=3,
        sleeper=delays.append,
    )

    assert result == "ok"
    assert delays == [5.0, 10.0]


def test_upload_validation_rejects_duplicate_ids() -> None:
    """Duplicate records must be rejected before a Hub upload."""

    dataset = pandas.DataFrame(
        [
            {"author_pub_id": "duplicate", "embedding": [0.1, 0.2]},
            {"author_pub_id": "duplicate", "embedding": [0.3, 0.4]},
        ]
    )

    with pytest.raises(RuntimeError, match="duplicate publication ids"):
        validate_publication_dataset(dataset)


def test_upload_validation_requires_exact_full_profile() -> None:
    """Full refreshes must contain exactly the discovered Scholar identifiers."""

    dataset = pandas.DataFrame(
        [{"author_pub_id": "current", "embedding": [0.1, 0.2]}]
    )

    with pytest.raises(RuntimeError, match="exactly match"):
        validate_publication_dataset(
            dataset,
            expected_publication_ids=["current", "missing"],
        )


def test_build_payload_does_not_mutate_scholar_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived bibliography fields must not leak into the source response."""

    publication = {
        "author_pub_id": "author:publication",
        "num_citations": 4,
        "bib": {
            "author": "Ada Lovelace and Grace Hopper",
            "title": "A New Design Method",
            "pub_year": 2026.0,
        },
    }
    def fake_bibtex(publication_for_bibtex: dict[str, object]) -> str:
        bibliography = publication_for_bibtex["bib"]
        assert isinstance(bibliography, dict)
        assert bibliography["pub_type"] == "article"
        assert bibliography["bib_id"].startswith("lovelace2026")
        return "@article{example}"

    monkeypatch.setattr(
        publication_sync.scholarly.scholarly,
        "bibtex",
        fake_bibtex,
    )

    payload = publication_sync.build_publication_payload(
        publication,
        [0.1, 0.2],
        datetime.date(2026, 7, 17),
    )

    assert "pub_type" not in publication["bib"]
    assert payload["bib_dict"]["bib_id"].startswith("lovelace2026")
    assert payload["Last Updated"] == "2026-07-17"


def test_missing_token_writes_failure_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secret configuration failures should leave an auditable status record."""

    status_path = tmp_path / "last-refresh.json"
    monkeypatch.delenv(publication_sync.HF_TOKEN_ENV, raising=False)

    exit_code = publication_sync.main(["--status-file", str(status_path)])
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert status["status"] == "failure"
    assert status["error_type"] == "RuntimeError"
    assert status["upload_performed"] is False
