"""Manually rebuild every publication embedding in the Hugging Face dataset."""

from __future__ import annotations

import logging
import os

from tqdm.auto import tqdm

from main import (
    HF_TOKEN_ENV,
    REPO_ID,
    embed_publication_text,
    load_specter2_model,
)

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Re-embed all rows and upload them as an explicit maintenance operation."""

    import datasets

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    hf_token = os.environ.get(HF_TOKEN_ENV)
    if not hf_token:
        raise RuntimeError(f"Missing {HF_TOKEN_ENV} environment variable")

    LOGGER.info("Loading %s", REPO_ID)
    dataset_df = datasets.load_dataset(REPO_ID, split="train").to_pandas()
    tokenizer, model = load_specter2_model()

    embeddings: list[list[float]] = []
    for bibliography in tqdm(dataset_df["bib_dict"], desc="Embedding publications"):
        bibliography = bibliography or {}
        embeddings.append(embed_publication_text(
            tokenizer,
            model,
            str(bibliography.get("title") or ""),
            str(bibliography.get("abstract") or ""),
        ))
    dataset_df["embedding"] = embeddings

    rebuilt_dataset = datasets.Dataset.from_pandas(
        dataset_df,
        preserve_index=False,
    )
    LOGGER.info("Uploading %d rebuilt rows", len(rebuilt_dataset))
    rebuilt_dataset.push_to_hub(
        REPO_ID,
        token=hf_token,
        commit_message="Rebuild all SPECTER2 embeddings",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
