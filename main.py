"""Synchronize Google Scholar publications to the Hugging Face dataset.

Normal runs discover new Scholar records and refresh a bounded set of stale
records. A manual full-refresh mode rebuilds every current Scholar record for
occasional baselining. Both modes atomically write a small status file so runs
remain observable.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, TypeVar

import pandas
import scholarly
from tqdm.auto import tqdm

REPO_ID = "ccm/publications"
AUTHOR_ID = "0P9w_S0AAAAJ"
HF_TOKEN_ENV = "HF_TOKEN"
STALE_PUBLICATION_REFRESH_LIMIT = 5
NEW_PUBLICATION_REFRESH_LIMIT = 25
SCHOLAR_REQUEST_DELAY_SECONDS = 2.0
SCHOLAR_RETRY_ATTEMPTS = 3
SPECTER_MODEL_NAME = "allenai/specter2_base"
SPECTER_MODEL_REVISION = "3447645e1def9117997203454fa4495937bfbd83"
SPECTER_ADAPTER_NAME = "allenai/specter2"
SPECTER_ADAPTER_REVISION = "2081559630a80fc5851d8f798a05ba81e9468089"
STATUS_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def configure_logging() -> None:
    """Configure concise structured console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def retry_operation(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = SCHOLAR_RETRY_ATTEMPTS,
    sleeper: Callable[[float], None] = time.sleep,
) -> T:
    """Run a Scholar operation with bounded exponential backoff."""

    if attempts < 1:
        raise ValueError("Retry attempts must be at least one")

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            delay = min(30.0, 5.0 * (2 ** (attempt - 1)))
            LOGGER.warning(
                "%s failed on attempt %d/%d; retrying in %.0f seconds",
                description,
                attempt,
                attempts,
                delay,
            )
            sleeper(delay)

    raise AssertionError("Retry loop exited unexpectedly")


def load_specter2_model() -> tuple[Any, Any]:
    """Load the SPECTER2 tokenizer and adapter-equipped model."""

    from adapters import AutoAdapterModel
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        SPECTER_MODEL_NAME,
        revision=SPECTER_MODEL_REVISION,
        trust_remote_code=False,
    )
    model = AutoAdapterModel.from_pretrained(
        SPECTER_MODEL_NAME,
        revision=SPECTER_MODEL_REVISION,
        trust_remote_code=False,
    )
    adapter_path = snapshot_download(
        SPECTER_ADAPTER_NAME,
        revision=SPECTER_ADAPTER_REVISION,
    )
    adapter_name = model.load_adapter(
        adapter_path,
        source="local",
        load_as="specter2",
        set_active=True,
    )
    model.set_active_adapters(adapter_name)
    model.eval()
    return tokenizer, model


def fetch_author_publications(
    author_id: str,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Retrieve lightweight publication metadata for an author."""

    LOGGER.info("Fetching publication list for author %s", author_id)
    author = retry_operation(
        lambda: scholarly.scholarly.search_author_id(author_id),
        description="Scholar author lookup",
        sleeper=sleeper,
    )
    if not author:
        raise RuntimeError(f"Author with id {author_id} not found")

    author = retry_operation(
        lambda: scholarly.scholarly.fill(author, sections=["publications"]),
        description="Scholar publication-list lookup",
        sleeper=sleeper,
    )
    publications = list(author.get("publications", []))
    if not publications:
        raise RuntimeError("Google Scholar returned no publications")
    LOGGER.info("Discovered %d publications from Google Scholar", len(publications))
    return publications


def ensure_last_updated_column(dataset_df: pandas.DataFrame) -> pandas.DataFrame:
    """Ensure the dataset contains a ``Last Updated`` column."""

    dataset_df = dataset_df.copy()
    if "Last Updated" not in dataset_df.columns:
        dataset_df["Last Updated"] = ""
    dataset_df["Last Updated"] = dataset_df["Last Updated"].fillna("")
    return dataset_df


def parse_last_updated(value: Any) -> datetime.date:
    """Parse ``Last Updated`` values into :class:`datetime.date`."""

    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            LOGGER.debug("Unparseable Last Updated value: %s", value)
    return datetime.date.min


@dataclass(frozen=True)
class PublicationRefreshPlan:
    """Plan describing which publications should be refreshed."""

    new_publication_ids: List[str]
    stale_publication_ids: List[str]
    metadata_only_publication_ids: List[str]

    @property
    def ordered_ids(self) -> List[str]:
        """Return publication identifiers in refresh order."""

        return [*self.new_publication_ids, *self.stale_publication_ids]


def determine_publications_to_refresh(
    remote_publications: Iterable[dict[str, Any]],
    existing_dataset: pandas.DataFrame,
    stale_refresh_limit: int,
    new_refresh_limit: int | None = None,
    full_refresh: bool = False,
) -> PublicationRefreshPlan:
    """Return a full or bounded incremental publication refresh plan."""

    if stale_refresh_limit < 0:
        raise ValueError("Stale refresh limit cannot be negative")
    if new_refresh_limit is not None and new_refresh_limit < 0:
        raise ValueError("New refresh limit cannot be negative")

    existing_dataset = ensure_last_updated_column(existing_dataset)
    existing_ids = set(
        existing_dataset.get("author_pub_id", pandas.Series(dtype=str))
        .dropna()
        .astype(str)
    )
    remote_ids = list(dict.fromkeys(
        str(publication["author_pub_id"])
        for publication in remote_publications
        if publication.get("author_pub_id")
    ))

    new_ids = [publication_id for publication_id in remote_ids if publication_id not in existing_ids]
    if full_refresh:
        existing_remote_ids = [
            publication_id for publication_id in remote_ids
            if publication_id in existing_ids
        ]
        LOGGER.info(
            "Full refresh plan contains %d new and %d existing publications",
            len(new_ids),
            len(existing_remote_ids),
        )
        return PublicationRefreshPlan(new_ids, [], existing_remote_ids)

    if new_refresh_limit is not None:
        new_ids = new_ids[:new_refresh_limit]

    working_df = existing_dataset.copy()
    if not working_df.empty and stale_refresh_limit > 0:
        working_df["author_pub_id"] = working_df["author_pub_id"].astype(str)
        working_df = working_df[working_df["author_pub_id"].isin(remote_ids)]
        working_df = working_df[~working_df["author_pub_id"].isin(new_ids)]
        working_df["_parsed_last_updated"] = working_df["Last Updated"].apply(
            parse_last_updated
        )
        stale_ids = (
            working_df.sort_values(
                by=["_parsed_last_updated", "author_pub_id"],
                ascending=[True, True],
            )
            .head(stale_refresh_limit)["author_pub_id"]
            .tolist()
        )
    else:
        stale_ids = []

    LOGGER.info(
        "Refresh plan contains %d new and %d stale publications",
        len(new_ids),
        len(stale_ids),
    )
    return PublicationRefreshPlan(new_ids, stale_ids, [])


def synchronize_listing_metadata(
    existing_dataset: pandas.DataFrame,
    remote_publications: Iterable[dict[str, Any]],
) -> pandas.DataFrame:
    """Prune absent records and update profile-list metadata for current ones."""

    existing_dataset = ensure_last_updated_column(existing_dataset)
    if "author_pub_id" not in existing_dataset.columns:
        return existing_dataset.iloc[0:0].copy()

    remote_lookup = {
        str(publication["author_pub_id"]): publication
        for publication in remote_publications
        if publication.get("author_pub_id")
    }
    synchronized = existing_dataset.copy()
    synchronized["author_pub_id"] = synchronized["author_pub_id"].astype(str)
    synchronized = synchronized[
        synchronized["author_pub_id"].isin(remote_lookup)
    ].copy()

    listing_fields = (
        "num_citations",
        "citedby_url",
        "cites_id",
        "pub_url",
        "url_related_articles",
    )
    for row_index, publication_id in synchronized["author_pub_id"].items():
        remote_publication = remote_lookup[publication_id]
        for field_name in listing_fields:
            value = remote_publication.get(field_name)
            if field_name in synchronized.columns and value is not None:
                synchronized.at[row_index, field_name] = value

    return synchronized.reset_index(drop=True)


def embed_publication_text(
    tokenizer: Any,
    model: Any,
    title: str,
    abstract: str,
) -> list[float]:
    """Generate a SPECTER2 embedding for the given title and abstract."""

    import torch

    model_inputs = tokenizer(
        (title or "") + tokenizer.sep_token + (abstract or ""),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )
    with torch.inference_mode():
        outputs = model(**model_inputs)
    embedding_tensor = outputs.last_hidden_state[0, 0, :].detach().cpu()
    return embedding_tensor.numpy().tolist()


def _normalized_publication_year(value: Any) -> str:
    """Return a stable year fragment for generated bibliography identifiers."""

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "unknown"


def build_publication_payload(
    publication: dict[str, Any],
    embedding: Sequence[float],
    updated_on: datetime.date,
) -> Dict[str, Any]:
    """Construct the dataset record for a publication without mutating input."""

    publication_bib = dict(publication.get("bib") or {})
    publication_bib["pub_type"] = "article"
    author_segment = str(publication_bib.get("author") or "").split(" and ")[0]
    first_author_last_name = author_segment.split(" ")[-1].lower() if author_segment else ""
    first_three_words = "".join(str(publication_bib.get("title") or "").split()[:3]).lower()
    year = _normalized_publication_year(publication_bib.get("pub_year"))
    publication_bib["bib_id"] = f"{first_author_last_name}{year}{first_three_words}"

    cites_per_year = {
        str(year_key): value
        for year_key, value in (publication.get("cites_per_year") or {}).items()
    }
    publication_for_bibtex = dict(publication)
    publication_for_bibtex["bib"] = publication_bib
    payload: Dict[str, Any] = {
        "bibtex": scholarly.scholarly.bibtex(publication_for_bibtex),
        "bib_dict": publication_bib,
        "author_pub_id": publication.get("author_pub_id"),
        "num_citations": int(publication.get("num_citations") or 0),
        "citedby_url": publication.get("citedby_url"),
        "cites_id": publication.get("cites_id"),
        "pub_url": publication.get("pub_url"),
        "url_related_articles": publication.get("url_related_articles"),
        "embedding": list(embedding),
        "Last Updated": updated_on.isoformat(),
    }
    payload.update(cites_per_year)
    return payload


def refresh_publications(
    target_ids: Sequence[str],
    remote_publications: Iterable[dict[str, Any]],
    *,
    delay_seconds: float = SCHOLAR_REQUEST_DELAY_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Download full metadata for target publications with throttling."""

    remote_lookup = {
        str(publication.get("author_pub_id")): publication
        for publication in remote_publications
        if publication.get("author_pub_id")
    }
    refreshed: list[dict[str, Any]] = []
    for index, publication_id in enumerate(target_ids):
        remote_stub = remote_lookup.get(publication_id)
        if remote_stub is None:
            LOGGER.warning("Publication %s is missing from the remote list", publication_id)
            continue
        LOGGER.info("Fetching full metadata for %s", publication_id)
        refreshed.append(retry_operation(
            lambda stub=dict(remote_stub): scholarly.scholarly.fill(stub),
            description=f"Scholar publication lookup ({publication_id})",
            sleeper=sleeper,
        ))
        if delay_seconds > 0 and index < len(target_ids) - 1:
            sleeper(delay_seconds)
    return refreshed


def merge_datasets(
    existing_dataset: pandas.DataFrame,
    new_records: Sequence[dict[str, Any]],
) -> pandas.DataFrame:
    """Merge refreshed records into the local dataset."""

    if not new_records:
        return existing_dataset.reset_index(drop=True)

    new_df = pandas.DataFrame.from_records(new_records)
    updated_dataset = pandas.concat(
        [
            existing_dataset[
                ~existing_dataset["author_pub_id"]
                .astype(str)
                .isin(new_df["author_pub_id"].astype(str))
            ],
            new_df,
        ],
        ignore_index=True,
        sort=False,
    )
    return updated_dataset.reset_index(drop=True)


def validate_publication_dataset(
    dataset_df: pandas.DataFrame,
    *,
    expected_publication_ids: Iterable[str] | None = None,
) -> None:
    """Reject malformed data before constructing or uploading a Hub dataset."""

    if dataset_df.empty:
        raise RuntimeError("Publication dataset is empty")
    if "author_pub_id" not in dataset_df.columns:
        raise RuntimeError("Publication dataset is missing author_pub_id")

    publication_ids = dataset_df["author_pub_id"]
    if publication_ids.isna().any() or (publication_ids.astype(str).str.len() == 0).any():
        raise RuntimeError("Publication dataset contains a missing publication id")
    normalized_ids = publication_ids.astype(str)
    if normalized_ids.duplicated().any():
        raise RuntimeError("Publication dataset contains duplicate publication ids")

    if expected_publication_ids is not None:
        expected_ids = {str(publication_id) for publication_id in expected_publication_ids}
        actual_ids = set(normalized_ids)
        if actual_ids != expected_ids:
            raise RuntimeError("Full refresh does not exactly match the Scholar profile")

    if "embedding" not in dataset_df.columns:
        raise RuntimeError("Publication dataset is missing embeddings")
    embedding_dimensions: set[int] = set()
    for embedding in dataset_df["embedding"]:
        try:
            embedding_array = pandas.array(embedding, dtype="float64").to_numpy()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Publication dataset contains a malformed embedding") from exc
        if embedding_array.ndim != 1 or embedding_array.size == 0:
            raise RuntimeError("Publication dataset contains an empty embedding")
        if not all(math.isfinite(float(value)) for value in embedding_array):
            raise RuntimeError("Publication dataset contains a non-finite embedding")
        embedding_dimensions.add(int(embedding_array.size))
    if len(embedding_dimensions) != 1:
        raise RuntimeError("Publication embeddings have inconsistent dimensions")


@dataclass(frozen=True)
class SyncResult:
    """Non-sensitive summary of one synchronization attempt."""

    discovered_publications: int
    new_publications: int
    stale_publications: int
    metadata_only_publications: int
    refreshed_publications: int
    dataset_rows: int
    upload_performed: bool
    dataset_commit: str | None
    refresh_mode: str
    snapshot_written: bool = False
    snapshot_sha256: str | None = None


def write_publication_snapshot(
    publication_dataset: Any,
    snapshot_path: Path,
    *,
    discovered_publications: int,
    new_publications: int,
    stale_publications: int,
    metadata_only_publications: int,
    refreshed_publications: int,
    refresh_mode: str,
) -> str:
    """Atomically write a validated Parquet snapshot and signed manifest."""

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    try:
        publication_dataset.to_parquet(str(temporary_path))
        snapshot_sha256 = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        os.replace(temporary_path, snapshot_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "success",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_id": REPO_ID,
        "author_id": AUTHOR_ID,
        "snapshot_file": snapshot_path.name,
        "snapshot_sha256": snapshot_sha256,
        "dataset_rows": len(publication_dataset),
        "discovered_publications": discovered_publications,
        "new_publications": new_publications,
        "stale_publications": stale_publications,
        "metadata_only_publications": metadata_only_publications,
        "refreshed_publications": refreshed_publications,
        "refresh_mode": refresh_mode,
    }
    manifest_path = snapshot_path.with_suffix(snapshot_path.suffix + ".manifest.json")
    write_status(manifest_path, manifest)
    return snapshot_sha256


def run_sync(
    *,
    hf_token: str | None,
    stale_limit: int,
    new_limit: int | None,
    request_delay: float,
    plan_only: bool,
    dry_run: bool,
    full_refresh: bool,
    snapshot_path: Path | None = None,
) -> SyncResult:
    """Execute a full or bounded Scholar-to-Hugging-Face synchronization."""

    import datasets

    LOGGER.info("Loading existing dataset from %s", REPO_ID)
    dataset = datasets.load_dataset(REPO_ID, split="train")
    dataset_df = ensure_last_updated_column(dataset.to_pandas())
    remote_publications = fetch_author_publications(AUTHOR_ID)
    refresh_plan = determine_publications_to_refresh(
        remote_publications,
        dataset_df,
        stale_refresh_limit=stale_limit,
        new_refresh_limit=new_limit,
        full_refresh=full_refresh,
    )
    refresh_mode = "full" if full_refresh else "incremental"

    if plan_only:
        return SyncResult(
            len(remote_publications),
            len(refresh_plan.new_publication_ids),
            len(refresh_plan.stale_publication_ids),
            len(refresh_plan.metadata_only_publication_ids),
            0,
            len(dataset_df),
            False,
            None,
            refresh_mode,
        )

    refreshed_publications = refresh_publications(
        refresh_plan.ordered_ids,
        remote_publications,
        delay_seconds=request_delay,
    )
    if refresh_plan.ordered_ids and not refreshed_publications:
        raise RuntimeError("No planned publications could be refreshed")
    if full_refresh and len(refreshed_publications) != len(refresh_plan.ordered_ids):
        raise RuntimeError("Full refresh did not retrieve every planned publication")

    refreshed_records: list[dict[str, Any]] = []
    if refreshed_publications:
        tokenizer, model = load_specter2_model()
        today = datetime.date.today()
        for publication in tqdm(refreshed_publications, desc="Embedding publications"):
            bibliography = publication.get("bib") or {}
            embedding = embed_publication_text(
                tokenizer,
                model,
                str(bibliography.get("title") or ""),
                str(bibliography.get("abstract") or ""),
            )
            refreshed_records.append(
                build_publication_payload(publication, embedding, today)
            )

    merge_base = (
        synchronize_listing_metadata(dataset_df, remote_publications)
        if full_refresh
        else dataset_df
    )
    merged_dataset = merge_datasets(merge_base, refreshed_records)
    current_year = datetime.date.today().year
    for year in range(2015, current_year + 1):
        column = str(year)
        if column not in merged_dataset.columns:
            merged_dataset[column] = None

    base_columns = [
        "bibtex",
        "bib_dict",
        "author_pub_id",
        "num_citations",
        "citedby_url",
        "cites_id",
        "pub_url",
        "url_related_articles",
        "embedding",
        "Last Updated",
    ]
    year_columns = sorted(
        (column for column in merged_dataset.columns if str(column).isdigit()),
        key=lambda column: int(str(column)),
    )
    ordered_columns = [
        column for column in [*base_columns, *year_columns]
        if column in merged_dataset.columns
    ]
    expected_publication_ids = (
        (
            str(publication["author_pub_id"])
            for publication in remote_publications
            if publication.get("author_pub_id")
        )
        if full_refresh
        else None
    )
    validate_publication_dataset(
        merged_dataset,
        expected_publication_ids=expected_publication_ids,
    )
    publication_dataset = datasets.Dataset.from_pandas(
        merged_dataset,
        preserve_index=False,
    ).select_columns(ordered_columns)

    snapshot_sha256 = None
    if snapshot_path is not None:
        snapshot_sha256 = write_publication_snapshot(
            publication_dataset,
            snapshot_path,
            discovered_publications=len(remote_publications),
            new_publications=len(refresh_plan.new_publication_ids),
            stale_publications=len(refresh_plan.stale_publication_ids),
            metadata_only_publications=len(refresh_plan.metadata_only_publication_ids),
            refreshed_publications=len(refreshed_records),
            refresh_mode=refresh_mode,
        )
        LOGGER.info("Wrote publication snapshot to %s", snapshot_path)

    dataset_commit = None
    upload_performed = False
    should_upload = bool(refreshed_records) or full_refresh
    if not dry_run and should_upload:
        if not hf_token:
            raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")
        LOGGER.info("Pushing %d dataset rows to Hugging Face Hub", len(publication_dataset))
        commit_info = publication_dataset.push_to_hub(
            REPO_ID,
            token=hf_token,
            commit_message="Refresh Google Scholar publications",
        )
        dataset_commit = getattr(commit_info, "oid", None)
        upload_performed = True
    elif dry_run:
        LOGGER.info("Dry run requested; skipping Hugging Face upload")

    return SyncResult(
        len(remote_publications),
        len(refresh_plan.new_publication_ids),
        len(refresh_plan.stale_publication_ids),
        len(refresh_plan.metadata_only_publication_ids),
        len(refreshed_records),
        len(publication_dataset),
        upload_performed,
        dataset_commit,
        refresh_mode,
        snapshot_path is not None,
        snapshot_sha256,
    )


def _workflow_url() -> str | None:
    """Build the current GitHub Actions run URL without exposing credentials."""

    server = os.environ.get("GITHUB_SERVER_URL")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return None


def write_status(path: Path, status: dict[str, Any]) -> None:
    """Atomically persist a non-sensitive refresh status record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", type=Path, default=Path("status/last-refresh.json"))
    parser.add_argument("--stale-limit", type=int, default=STALE_PUBLICATION_REFRESH_LIMIT)
    parser.add_argument("--new-limit", type=int, default=NEW_PUBLICATION_REFRESH_LIMIT)
    parser.add_argument("--request-delay", type=float, default=SCHOLAR_REQUEST_DELAY_SECONDS)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        help="Write the validated merged dataset to this Parquet snapshot",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Rebuild every current Scholar record instead of using incremental limits",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that always records a bounded status result."""

    args = _parse_args(argv)
    configure_logging()
    started_at = datetime.datetime.now(datetime.timezone.utc)
    status: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "dataset_id": REPO_ID,
        "author_id": AUTHOR_ID,
        "workflow_url": _workflow_url(),
        "refresh_mode": "full" if args.full_refresh else "incremental",
    }

    try:
        hf_token = os.environ.get(HF_TOKEN_ENV)
        if not args.plan_only and not args.dry_run and not hf_token:
            raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")
        result = run_sync(
            hf_token=hf_token,
            stale_limit=args.stale_limit,
            new_limit=args.new_limit,
            request_delay=args.request_delay,
            plan_only=args.plan_only,
            dry_run=args.dry_run,
            full_refresh=args.full_refresh,
            snapshot_path=args.snapshot_path,
        )
        status.update({
            "status": "success",
            "discovered_publications": result.discovered_publications,
            "new_publications": result.new_publications,
            "stale_publications": result.stale_publications,
            "metadata_only_publications": result.metadata_only_publications,
            "refreshed_publications": result.refreshed_publications,
            "dataset_rows": result.dataset_rows,
            "upload_performed": result.upload_performed,
            "dataset_commit": result.dataset_commit,
            "refresh_mode": result.refresh_mode,
            "snapshot_written": result.snapshot_written,
            "snapshot_sha256": result.snapshot_sha256,
            "error_type": None,
        })
        exit_code = 0
    except Exception as exc:
        LOGGER.exception("Publication refresh failed")
        status.update({
            "status": "failure",
            "upload_performed": False,
            "error_type": type(exc).__name__,
        })
        exit_code = 1

    status["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_status(args.status_file, status)
    LOGGER.info("Wrote refresh status to %s", args.status_file)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
