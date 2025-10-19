"""Unit tests for the publication refresh strategy."""

import datetime
import sys
from pathlib import Path

import pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import determine_publications_to_refresh, ensure_last_updated_column


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

    selection = determine_publications_to_refresh(
        remote_publications, existing_dataset, 2
    )

    assert selection == ["pub2", "pub3"]


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

    selection = determine_publications_to_refresh(
        remote_publications, existing_dataset, 2
    )

    assert selection == ["pub2", "pub3"]


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
