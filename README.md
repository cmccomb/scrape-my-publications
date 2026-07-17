# scrape-my-publications

This repository synchronizes Chris McComb's Google Scholar profile to the
[`ccm/publications`](https://huggingface.co/datasets/ccm/publications) dataset.
The website repository then turns that dataset into the interactive publication
graph at [cmccomb.com](https://cmccomb.com/).

## Monthly refresh

The scheduled workflow runs on the first day of each month and:

1. retrieves the current list of Scholar publication identifiers;
2. adds up to 25 newly discovered publications;
3. refreshes five of the stalest existing publications;
4. waits between detail requests and retries transient Scholar failures;
5. generates SPECTER2 embeddings only for refreshed records;
6. pushes the merged dataset to Hugging Face; and
7. commits a non-sensitive status record to `status/last-refresh.json`.

The status commit provides a durable success/failure audit trail and keeps the
public repository active so GitHub does not silently disable scheduled runs.
The website graph refresh runs separately on the second day of each month,
after this dataset update has had time to finish.

The main workflow also exposes a manual `full_refresh` input. Use it for an
occasional clean baseline. It mirrors the current Scholar profile, drops
records no longer present, updates citation totals for every retained record,
and fetches full details plus embeddings for newly discovered works. Existing
embeddings are preserved. The separate full-dataset re-embedding workflow is
also manual-only. Normal monthly updates should use the incremental refresh.

## Local checks

Install the pinned test environment and run the unit tests:

```bash
python -m pip install --requirement requirements-dev.txt
python -m pytest
```

Inspect the current refresh plan without downloading publication details,
loading the embedding model, or uploading data:

```bash
python main.py --plan-only --status-file status/local-plan.json
```

Run a full refresh without uploading it:

```bash
python main.py --full-refresh --dry-run --status-file status/local-dry-run.json
```

Production uploads read the Hugging Face credential from the `HF_TOKEN`
environment variable. Tokens are never accepted as command-line arguments or
written to status files.

The AllenAI SPECTER2 base model and adapter are pinned to immutable Hugging
Face commit SHAs. Remote model code is disabled, and automation never accepts
an arbitrary model repository or checkpoint.
