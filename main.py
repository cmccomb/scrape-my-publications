import scholarly
from tqdm.auto import tqdm
import sentence_transformers
import datasets
import sys

# Get author info from Google Scholar
author = scholarly.scholarly.fill(scholarly.scholarly.search_author_id("0P9w_S0AAAAJ"))

# Load embedding model
model = sentence_transformers.SentenceTransformer("allenai-specter")

# Declare blank list to append to
publication_info = []

# Iterate through publications and fill
for i in tqdm(range(len(author["publications"]))):
    this_pub = scholarly.scholarly.fill(author["publications"][i])
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
            "embedding": model.encode(
                this_pub["bib"]["title"] + " " + str(this_pub["bib"].get("abstract"))
            ),
        }
    )

# Convert to a dataset and upload to hugginface
dataset = datasets.Dataset.from_list(publication_info)
dataset.push_to_hub("ccm/publications", token=sys.argv[1])
