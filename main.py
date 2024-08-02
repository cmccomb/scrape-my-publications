import sys  # Used to read token argument from command line

import datasets  # Used to upload to huggingface
import pandas  # Used to convert to a dataset
import scholarly  # Used to get author info from Google Scholar
import sentence_transformers  # Used to embed the publication text
from tqdm.auto import tqdm  # Used to show progress bar

# Get API token from command line
HF_TOKEN = sys.argv[1]

# Get author info from Google Scholar
author = scholarly.scholarly.fill(scholarly.scholarly.search_author_id("0P9w_S0AAAAJ"))

# Load embedding model
model = sentence_transformers.SentenceTransformer("allenai-specter")

# Declare blank list to append to
publication_data = []

# Iterate through publications and fill
for i in tqdm(range(len(author["publications"]))):
    # Fill the publication info
    publication = scholarly.scholarly.fill(author["publications"][i])

    # Embed the publication text
    embedding_vector = model.encode(
        publication["bib"]["title"] + " " + str(publication["bib"].get("abstract"))
    )

    publication["bib"].update({"pub_type": "article"})
    publication["bib"].update(
        {
            "bib_id": author["publications"][i]["bib"]["author"]
            .split(" and ")[0]
            .split(" ")[-1]
            .lower()
            + str(author["publications"][i]["bib"].get("pub_year", 0))
            + "".join(author["publications"][i]["bib"].get("title", "").split(" ")[:2])
        }
    )

    # Append data to the list
    publication_data.append(
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

# Convert to a dataset and upload to huggingface. Converting to pandas and then to dataset avoids some weird errors
publication_dataset = datasets.Dataset.from_pandas(
    pandas.DataFrame.from_dict(publication_data)
)
publication_dataset.push_to_hub("ccm/publications", token=HF_TOKEN)
