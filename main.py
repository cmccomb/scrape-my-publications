"""
This script automates the retrieval, embedding, and cloud storage of academic publication data
for a specified Google Scholar author profile. Designed to keep a Hugging Face dataset
(ccm/publications) current, it pulls new publications and refreshes a subset of existing ones,
embedding each using the SPECTER2 model.

Main Features:
---------------
- Fetches publication metadata from Google Scholar via the `scholarly` library
- Detects and processes new publications or stale entries (based on update timestamps)
- Embeds title + abstract using the SPECTER2 model (via Hugging Face adapters)
- Formats metadata including BibTeX, citation counts, and unique BibTeX-style IDs
- Outputs a merged dataset and pushes it to the Hugging Face Hub

Inputs:
-------
- Hugging Face API token (via command line argument)
- Author ID for Google Scholar (hardcoded)
- Existing dataset hosted at `ccm/publications` on Hugging Face

Outputs:
--------
- Updated Hugging Face dataset with:
    * BibTeX citation data
    * Author and citation metadata
    * Dense embeddings for text
    * Yearly citation counts
    * Last update date

Usage:
------
python update_publication_embeddings.py <HF_API_TOKEN>
"""

import datetime  # Used to get today's date
import sys  # Used to read token argument from command line

import datasets  # Used to upload to huggingface
import pandas  # Used to convert to a dataset
import scholarly  # Used to get author info from Google Scholar
from adapters import AutoAdapterModel
from tqdm.auto import tqdm  # Used to show progress bar
from transformers import AutoTokenizer

# Constants for settings
HF_TOKEN = sys.argv[1]  # Get API token from command line
REPO_ID = "ccm/publications"  # Huggingface repo ID
AUTHOR_ID = "0P9w_S0AAAAJ"  # Author ID for Google Scholar
MAX_UPDATED_PUBLICATIONS = 2  # Max number of old publications to update at a time

# Load embedding model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Download the repo from REPO_ID using datasets
dataset = datasets.load_dataset(REPO_ID, split="train")

# Get the publications ids from in the current dataset
publications_in_current_dataset = dataset.to_pandas()["author_pub_id"].values

# Select MAX_UPDATED_PUBLICATIONS to update, based on the oldest update date
publications_to_update = (
    dataset.sort("Last Updated")
    .select(range(min(MAX_UPDATED_PUBLICATIONS, len(dataset))))
    .to_pandas()["author_pub_id"]
    .values
).tolist()

# Get author info from Google Scholar and then publication IDs
author = scholarly.scholarly.fill(scholarly.scholarly.search_author_id(AUTHOR_ID))
publications_from_google_scholar = pandas.DataFrame.from_dict(author["publications"])[
    "author_pub_id"
].values

# Find new publications
new_publication_ids = [
    pub_id
    for pub_id in publications_from_google_scholar
    if pub_id not in publications_in_current_dataset
] + publications_to_update
new_publications = [
    next(
        (d for d in author["publications"] if d.get("author_pub_id") == new_pub_id),
        None,
    )
    for new_pub_id in new_publication_ids
]

new_publication_data = []

# Iterate through publications and fill
for i in tqdm(range(len(new_publications)), desc="Processing new publications"):

    # Fill the publication info
    publication = scholarly.scholarly.fill(new_publications[i])

    # Embed the title and abstract
    embedding_vector = (
        model(
            **tokenizer(
                (publication["bib"].get("title") or "")
                + tokenizer.sep_token
                + (publication["bib"].get("abstract") or ""),
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
        )
        .last_hidden_state[:, 0, :]
        .detach()
        .numpy()[0]
    )

    # Update publication info in order to format bibtex well
    publication["bib"].update({"pub_type": "article"})
    first_authors_last_name = (
        publication["bib"]["author"].split(" and ")[0].split(" ")[-1].lower()
    )
    first_three_words_from_title = "".join(
        publication["bib"].get("title", "").split(" ")[:3]
    ).lower()
    year = str(publication["bib"].get("pub_year", 0))
    publication["bib"].update(
        {"bib_id": first_authors_last_name + year + first_three_words_from_title}
    )

    # Append data to the list
    new_publication_data.append(
        {
            "bibtex": scholarly.scholarly.bibtex(publication),
            "bib_dict": publication["bib"],
            "author_pub_id": publication.get("author_pub_id"),
            "num_citations": publication.get("num_citations"),
            "citedby_url": publication.get("citedby_url"),
            "cites_id": publication.get("cites_id"),
            "pub_url": publication.get("pub_url"),
            "url_related_articles": publication.get("url_related_articles"),
            **{str(k): v for k, v in publication["cites_per_year"].items()},
            "embedding": embedding_vector,
            "Last Updated": datetime.date.today().strftime("%Y-%m-%d"),
        }
    )

# Convert to a dataset. Converting to pandas and then to dataset avoids some weird errors
new_table = pandas.DataFrame.from_dict(new_publication_data)
new_table.rename(columns={2024: "2024", 2025: "2025"}, inplace=True)
new_table = pandas.concat(
    [dataset.to_pandas(), new_table], ignore_index=True, sort=False
)

# Remove any duplicated entries in author_pub_id
new_table = new_table.drop_duplicates(
    subset=["author_pub_id"], keep="last"
).reset_index(drop=True)

# Remove level_0 column if it exists
if "level_0" in new_table.columns:
    new_table = new_table.drop(columns=["level_0"])

# Make it a dataset and upload to huggingface, clean it up, and upload it to huggingface
publication_dataset = datasets.Dataset.from_pandas(new_table)
publication_dataset = publication_dataset.select_columns(
    [
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
    + [str(year) for year in range(2015, datetime.date.today().year + 1)]
)
publication_dataset.push_to_hub(REPO_ID, token=HF_TOKEN)
