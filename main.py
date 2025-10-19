"""scrape-my-publications main workflow.

This module updates the ``ccm/publications`` dataset by retrieving metadata
for publications from Google Scholar, creating SPECTER2 embeddings, and pushing
results to the Hugging Face Hub.  The workflow is intentionally incremental to
avoid repeatedly downloading every publication, which can trigger throttling
from Google Scholar.

The update logic operates in two phases:

1. Perform a lightweight metadata sync to discover publication identifiers.
2. If new publications exist, fully download ("fill") and embed those entries.
   Otherwise refresh a small batch of the stalest publications.

The file exposes ``main`` so the behavior can be unit-tested without triggering
network calls on import.
"""

from __future__ import annotations

import datetime
import logging
import sys
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List

import datasets
import pandas
import scholarly
import torch
from adapters import AutoAdapterModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer

HF_TOKEN_INDEX = 1
REPO_ID = "ccm/publications"
AUTHOR_ID = "0P9w_S0AAAAJ"
MAX_UPDATED_PUBLICATIONS = 2
SPECTER_MODEL_NAME = "allenai/specter2_base"
SPECTER_ADAPTER_NAME = "allenai/specter2"
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure a simple console logger."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_specter2_model() -> tuple[AutoTokenizer, AutoAdapterModel]:
    """Load the SPECTER2 tokenizer and adapter-equipped model."""

    tokenizer = AutoTokenizer.from_pretrained(SPECTER_MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(SPECTER_MODEL_NAME)
    model.load_adapter(
        SPECTER_ADAPTER_NAME,
        source="hf",
        load_as="specter2",
        set_active=True,
    )
    return tokenizer, model


def fetch_author_publications(author_id: str) -> list[dict[str, Any]]:
    """Retrieve lightweight publication metadata for an author."""

    LOGGER.info("Fetching publication list for author %s", author_id)
    author = scholarly.scholarly.search_author_id(author_id)
    if not author:
        msg = f"Author with id {author_id} not found"
        raise RuntimeError(msg)

    try:
        author = scholarly.scholarly.fill(author, sections=["publications"])
    except TypeError:
        # Older scholarly versions do not accept ``sections``.
        author = scholarly.scholarly.fill(author)

    publications: list[dict[str, Any]] = list(author.get("publications", []))
    LOGGER.info("Discovered %d publications from Google Scholar", len(publications))
    return publications


def ensure_last_updated_column(dataset_df: pandas.DataFrame) -> pandas.DataFrame:
    """Ensure the dataset contains a ``Last Updated`` column."""

    if "Last Updated" not in dataset_df.columns:
        dataset_df["Last Updated"] = ""
    dataset_df["Last Updated"] = dataset_df["Last Updated"].fillna("")
    return dataset_df


def parse_last_updated(value: Any) -> datetime.date:
    """Parse ``Last Updated`` values into :class:`datetime.date`."""

    if isinstance(value, datetime.date):
        return value
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, str) and value:
        try:
            return datetime.date.fromisoformat(value)
        except ValueError:
            LOGGER.debug("Unparseable Last Updated value: %s", value)
    return datetime.date.min


def determine_publications_to_refresh(
    remote_publications: Iterable[dict[str, Any]],
    existing_dataset: pandas.DataFrame,
    max_refresh: int,
) -> list[str]:
    """Return publication identifiers that should be refreshed."""

    existing_dataset = ensure_last_updated_column(existing_dataset.copy())
    existing_ids: set[str] = set(
        existing_dataset.get("author_pub_id", pandas.Series(dtype=str))
        .dropna()
        .astype(str)
    )
    remote_ids: list[str] = [
        publication.get("author_pub_id")
        for publication in remote_publications
        if publication.get("author_pub_id")
    ]

    new_ids = [pub_id for pub_id in remote_ids if pub_id not in existing_ids]
    if new_ids:
        LOGGER.info("Identified %d new publications", len(new_ids))
        return new_ids

    LOGGER.info("No new publications discovered; refreshing stale entries")
    working_df = existing_dataset.copy()
    working_df["_parsed_last_updated"] = working_df["Last Updated"].apply(
        parse_last_updated
    )
    working_df["author_pub_id"] = working_df["author_pub_id"].astype(str)

    refreshed_ids = (
        working_df.sort_values(
            by=["_parsed_last_updated", "author_pub_id"],
            ascending=[True, True],
        )
        .head(max_refresh)["author_pub_id"]
        .tolist()
    )
    return refreshed_ids


def embed_publication_text(
    tokenizer: AutoTokenizer,
    model: AutoAdapterModel,
    title: str,
    abstract: str,
) -> list[float]:
    """Generate a SPECTER2 embedding for the given title + abstract."""

    model_inputs = tokenizer(
        (title or "") + tokenizer.sep_token + (abstract or ""),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_token_type_ids=False,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**model_inputs)
    embedding_tensor = outputs.last_hidden_state[0, 0, :].detach().cpu()
    return embedding_tensor.numpy().tolist()


def build_publication_payload(
    publication: dict[str, Any],
    embedding: Sequence[float],
    updated_on: datetime.date,
) -> Dict[str, Any]:
    """Construct the dataset record for a publication."""

    publication_bib = publication.get("bib", {})
    publication_bib.update({"pub_type": "article"})
    author_segment = publication_bib.get("author", "").split(" and ")[0]
    first_authors_last_name = (
        author_segment.split(" ")[-1].lower() if author_segment else ""
    )
    first_three_words = "".join(publication_bib.get("title", "").split(" ")[:3]).lower()
    year = str(publication_bib.get("pub_year", 0))
    publication_bib.update(
        {"bib_id": f"{first_authors_last_name}{year}{first_three_words}"}
    )

    cites_per_year = {
        str(year_key): value
        for year_key, value in publication.get("cites_per_year", {}).items()
    }

    payload: Dict[str, Any] = {
        "bibtex": scholarly.scholarly.bibtex(publication),
        "bib_dict": publication_bib,
        "author_pub_id": publication.get("author_pub_id"),
        "num_citations": publication.get("num_citations"),
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
) -> list[dict[str, Any]]:
    """Download full metadata for the target publications."""

    remote_lookup = {
        publication.get("author_pub_id"): publication
        for publication in remote_publications
    }
    refreshed: list[dict[str, Any]] = []
    for publication_id in target_ids:
        remote_stub = remote_lookup.get(publication_id)
        if remote_stub is None:
            LOGGER.warning("Publication %s missing from remote list", publication_id)
            continue
        LOGGER.info("Fetching full metadata for %s", publication_id)
        refreshed.append(scholarly.scholarly.fill(dict(remote_stub)))
    return refreshed


def merge_datasets(
    existing_dataset: pandas.DataFrame,
    new_records: Sequence[dict[str, Any]],
) -> pandas.DataFrame:
    """Merge refreshed records into the local dataset."""

    if not new_records:
        LOGGER.info("No publication updates produced; dataset will remain unchanged")
        return existing_dataset

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


def main(argv: Sequence[str] | None = None) -> None:
    """Entrypoint for the update workflow."""

    argv = list(argv or sys.argv)
    if len(argv) <= HF_TOKEN_INDEX:
        raise SystemExit("Missing Hugging Face token argument")

    configure_logging()
    hf_token = argv[HF_TOKEN_INDEX]

    LOGGER.info("Loading existing dataset from %s", REPO_ID)
    dataset = datasets.load_dataset(REPO_ID, split="train")
    dataset_df = ensure_last_updated_column(dataset.to_pandas())

    remote_publications = fetch_author_publications(AUTHOR_ID)
    target_ids = determine_publications_to_refresh(
        remote_publications=remote_publications,
        existing_dataset=dataset_df,
        max_refresh=MAX_UPDATED_PUBLICATIONS,
    )

    tokenizer, model = load_specter2_model()
    refreshed_publications = refresh_publications(target_ids, remote_publications)

    today = datetime.date.today()
    refreshed_records: list[dict[str, Any]] = []
    for publication in tqdm(refreshed_publications, desc="Processing publications"):
        embedding = embed_publication_text(
            tokenizer,
            model,
            publication.get("bib", {}).get("title", ""),
            publication.get("bib", {}).get("abstract", ""),
        )
        refreshed_records.append(
            build_publication_payload(
                publication=publication, embedding=embedding, updated_on=today
            )
        )

    merged_dataset = merge_datasets(dataset_df, refreshed_records)

    # Ensure year columns exist for the push. Create placeholders if missing.
    current_year = datetime.date.today().year
    for year in range(2015, current_year + 1):
        column = str(year)
        if column not in merged_dataset.columns:
            merged_dataset[column] = pandas.NA

    publication_dataset = datasets.Dataset.from_pandas(merged_dataset)
    ordered_columns = [
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
    ] + [str(year) for year in range(2015, current_year + 1)]
    available_columns = [
        col for col in ordered_columns if col in publication_dataset.column_names
    ]
    publication_dataset = publication_dataset.select_columns(available_columns)

    LOGGER.info("Pushing dataset updates to Hugging Face Hub")
    publication_dataset.push_to_hub(REPO_ID, token=hf_token)


if __name__ == "__main__":  # pragma: no cover
    main()
