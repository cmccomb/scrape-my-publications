"""Validate and upload a locally collected publication snapshot."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import datasets  # type: ignore[import-untyped]
import pandas  # type: ignore[import-untyped]

from main import (
    AUTHOR_ID,
    HF_TOKEN_ENV,
    REPO_ID,
    SNAPSHOT_SCHEMA_VERSION,
    STATUS_SCHEMA_VERSION,
    _workflow_url,
    validate_publication_dataset,
    write_status,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_SNAPSHOT_AGE_DAYS = 7


def load_and_validate_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    max_age_days: int = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
) -> tuple[pandas.DataFrame, dict[str, Any]]:
    """Return a snapshot only after its provenance and contents validate."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "created_at_utc",
        "dataset_id",
        "author_id",
        "snapshot_file",
        "snapshot_sha256",
        "dataset_rows",
        "discovered_publications",
        "refresh_mode",
    }
    missing = required.difference(manifest)
    if missing:
        raise RuntimeError(f"Snapshot manifest is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise RuntimeError("Snapshot manifest schema is unsupported")
    if manifest["status"] != "success":
        raise RuntimeError("Snapshot collector did not report success")
    if manifest["dataset_id"] != REPO_ID or manifest["author_id"] != AUTHOR_ID:
        raise RuntimeError("Snapshot provenance does not match this publication dataset")
    if manifest["snapshot_file"] != snapshot_path.name:
        raise RuntimeError("Snapshot filename does not match its manifest")

    created_at = datetime.datetime.fromisoformat(str(manifest["created_at_utc"]))
    if created_at.tzinfo is None:
        raise RuntimeError("Snapshot timestamp must include a timezone")
    age = datetime.datetime.now(datetime.timezone.utc) - created_at
    if age < datetime.timedelta(minutes=-5):
        raise RuntimeError("Snapshot timestamp is unexpectedly in the future")
    if age > datetime.timedelta(days=max_age_days):
        raise RuntimeError("Snapshot is too old to upload")

    actual_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(actual_sha256, str(manifest["snapshot_sha256"])):
        raise RuntimeError("Snapshot checksum does not match its manifest")

    snapshot_df = pandas.read_parquet(snapshot_path)
    validate_publication_dataset(snapshot_df)
    if len(snapshot_df) != int(manifest["dataset_rows"]):
        raise RuntimeError("Snapshot row count does not match its manifest")
    if (
        manifest["refresh_mode"] == "full"
        and len(snapshot_df) != int(manifest["discovered_publications"])
    ):
        raise RuntimeError("Snapshot does not exactly mirror the discovered Scholar profile")
    return snapshot_df, manifest


def upload_snapshot(
    snapshot_path: Path,
    manifest_path: Path,
    *,
    hf_token: str,
    max_age_days: int = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
) -> tuple[str | None, dict[str, Any]]:
    """Upload one validated snapshot and return the Hub commit."""

    snapshot_df, manifest = load_and_validate_snapshot(
        snapshot_path,
        manifest_path,
        max_age_days=max_age_days,
    )
    publication_dataset = datasets.Dataset.from_pandas(
        snapshot_df,
        preserve_index=False,
    ).select_columns(list(snapshot_df.columns))
    commit_info = publication_dataset.push_to_hub(
        REPO_ID,
        token=hf_token,
        commit_message="Upload validated Google Scholar snapshot",
    )
    return getattr(commit_info, "oid", None), manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with a durable, non-sensitive status record."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = _parse_args(argv)
    started_at = datetime.datetime.now(datetime.timezone.utc)
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "dataset_id": REPO_ID,
        "author_id": AUTHOR_ID,
        "workflow_url": _workflow_url(),
        "upload_performed": False,
    }
    try:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if not hf_token:
            raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")
        dataset_commit, manifest = upload_snapshot(
            args.snapshot,
            args.manifest,
            hf_token=hf_token,
            max_age_days=args.max_age_days,
        )
        status.update({
            "status": "success",
            "dataset_commit": dataset_commit,
            "dataset_rows": manifest["dataset_rows"],
            "discovered_publications": manifest["discovered_publications"],
            "new_publications": manifest.get("new_publications", 0),
            "stale_publications": manifest.get("stale_publications", 0),
            "metadata_only_publications": manifest.get(
                "metadata_only_publications", 0
            ),
            "refreshed_publications": manifest.get("refreshed_publications", 0),
            "refresh_mode": manifest["refresh_mode"],
            "snapshot_sha256": manifest["snapshot_sha256"],
            "upload_performed": True,
            "error_type": None,
        })
        exit_code = 0
    except Exception as exc:
        LOGGER.exception("Snapshot upload failed")
        status.update({
            "status": "failure",
            "error_type": type(exc).__name__,
        })
        exit_code = 1

    status["finished_at_utc"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    write_status(args.status_file, status)
    LOGGER.info("Wrote upload status to %s", args.status_file)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
