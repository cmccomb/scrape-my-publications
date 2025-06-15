import sys  # Used to read token argument from command line

import datasets  # Used to upload to huggingface
import pandas  # Used to convert to a dataset
from adapters import AutoAdapterModel
from tqdm.auto import tqdm  # Used to show progress bar
from transformers import AutoTokenizer

# Constants for settings
HF_TOKEN = sys.argv[1]  # Get API token from command line
REPO_ID = "ccm/publications"  # Huggingface repo ID
AUTHOR_ID = "0P9w_S0AAAAJ"  # Author ID for Google Scholar
MAX_UPDATED_PUBLICATIONS = 10  # Max number of old publications to update at a time

# Load embedding model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

# Download the repo from REPO_ID using datasets
dataset = (
    datasets.load_dataset(REPO_ID, split="train").to_pandas().to_dict(orient="records")
)

# Iterate through publications and fill
for i in tqdm(range(len(dataset)), desc="Processing new publications"):

    # Embed the title and abstract
    embedding_vector = (
        model(
            **tokenizer(
                (dataset[i]["bib_dict"].get("title") or "")
                + tokenizer.sep_token
                + (dataset[i]["bib_dict"].get("abstract") or ""),
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

    dataset[i]["embedding"] = embedding_vector

# Upload to huggingface
dataset = datasets.Dataset.from_pandas(pandas.DataFrame.from_dict(dataset))
dataset.push_to_hub(REPO_ID, token=HF_TOKEN)
