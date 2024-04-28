import sys  # Used to read token argument from command line

import datasets  # Used to upload to huggingface
import pandas  # Used to convert to a dataset
import scholarly  # Used to get author info from Google Scholar
import sentence_transformers  # Used to embed the publication text
from tqdm.auto import tqdm  # Used to show progress bar

# Get author info from Google Scholar
author = scholarly.scholarly.fill(scholarly.scholarly.search_author_id("0P9w_S0AAAAJ"))

# Load embedding model
model = sentence_transformers.SentenceTransformer("allenai-specter")

# Declare blank list to append to
publication_info = []

# Iterate through publications and fill
for i in tqdm(range(len(author["publications"]))):
    this_pub = scholarly.scholarly.fill(author["publications"][i])
    vector = model.encode(
        this_pub["bib"]["title"] + " " + str(this_pub["bib"].get("abstract"))
    )
    publication_info.append(
        {
            **this_pub["bib"],
            "author_pub_id": this_pub.get("author_pub_id"),
            "num_citations": this_pub.get("num_citations"),
            "citedby_url": this_pub.get("citedby_url"),
            "cites_id": this_pub.get("cites_id"),
            "pub_url": this_pub.get("pub_url"),
            "url_related_articles": this_pub.get("url_related_articles"),
            **this_pub["cites_per_year"],
            "embedding": vector,
        }
    )

# Convert to a dataset and upload to huggingface
dataset = datasets.Dataset.from_pandas(pandas.DataFrame.from_dict(publication_info))
dataset.push_to_hub("ccm/publications", token=sys.argv[1])
