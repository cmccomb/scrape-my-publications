import random  # Used for random sleep
import sys  # Used to read token argument from command line
import time  # Used for random sleep

import datasets  # Used to upload to huggingface
import pandas  # Used to convert to a dataset
import scholarly  # Used to get author info from Google Scholar
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm.auto import tqdm  # Used to show progress bar

# Settings
HF_TOKEN = sys.argv[1]  # Get API token from command line
REPO_ID = "ccm/publications"  # Huggingface repo ID
AUTHOR_ID = "0P9w_S0AAAAJ"  # Author ID for Google Scholar
MAX_PUBLICATIONS = 100  # Max number of publications to process per run

# Load embedding model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Download the repo from REPO_ID using datasets
dataset = datasets.load_dataset(REPO_ID, split="train")

# Get author info from Google Scholar
author = scholarly.scholarly.fill(scholarly.scholarly.search_author_id(AUTHOR_ID))

# Find new publications
publications_in_current_dataset = dataset.to_pandas()["author_pub_id"].values
publications_from_google_scholar = pandas.DataFrame.from_dict(author["publications"])[
    "author_pub_id"
].values
new_publication_ids = [
    pub_id
    for pub_id in publications_from_google_scholar
    if pub_id not in publications_in_current_dataset
]
new_publications = [
    next(
        (d for d in author["publications"] if d.get("author_pub_id") == new_pub_id),
        None,
    )
    for new_pub_id in new_publication_ids
]

# Declare blank list to append to
new_publication_data = []

# Iterate through publications and fill
for i in tqdm(range(len(new_publications))):
    print(new_publications[i])

    # Fill the publication info
    publication = scholarly.scholarly.fill(new_publications[i])

    # Embed the title and abstract
    embedding_vector = (
        model(
            **tokenizer(
                publication["bib"]["title"]
                + tokenizer.sep_token
                + publication["bib"]["abstract"],
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
            **publication["cites_per_year"],
            "embedding": embedding_vector,
        }
    )

# Convert to a dataset. Converting to pandas and then to dataset avoids some weird errors
new_table = pandas.DataFrame.from_dict(new_publication_data)
new_table.rename(columns={2024: "2024", 2025: "2025"}, inplace=True)

# Concatenate the DataFrames
publication_dataset = datasets.Dataset.from_pandas(
    pandas.concat(
        [dataset.to_pandas(), new_table], ignore_index=True, sort=False
    ).reset_index()
)
# Upload to huggingface
publication_dataset.push_to_hub(REPO_ID, token=HF_TOKEN)
