# scrape-my-publications

## Overview

`scrape-my-publications` synchronizes the [`ccm/publications`](https://huggingface.co/datasets/ccm/publications)
dataset with metadata from Google Scholar. The workflow keeps the dataset up to
date by discovering new publications for a target author, generating SPECTER2
embeddings, and periodically refreshing existing entries.

## Update workflow

The automated job performs the following steps:

1. **Author sync** – perform a lightweight `scholarly.fill` on the author to
   retrieve the current list of publication identifiers.
2. **New publication ingest** – fully fetch metadata for any new publications
   that are not yet present in the dataset and add them to the corpus.
3. **Background refresh** – update a small batch of the stalest publications on
   each run (currently limited to ten items) so historical records gradually
   receive fresh metadata.
4. **Embedding generation** – build SPECTER2 embeddings for every refreshed
   publication.
5. **Dataset push** – merge the refreshed records back into the existing
   dataset and push the results to the Hugging Face Hub.

## Development

Install dependencies with `pip install -r requirements.txt` and run `pytest`
to execute the unit tests. The codebase targets Python 3.11 and expects the
standard tooling defined in `requirements.txt` to be available.
