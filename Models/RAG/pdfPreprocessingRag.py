import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, DPRContextEncoder
import torch
import time
import datetime
import math
import numpy as np
import faiss
from datasets import Dataset
import pandas as pd
import apiKeys


def pdf_to_html(link):
    api_url = "https://debi-api.azurewebsites.net/ChatDoc/GetHtmlFromUrl"

    # bearer token from Postman
    BEARER_TOKEN = apiKeys.apiBearerToken
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    params = {"url": link, "responseAsFileUrl": "false"}
    response = requests.get(api_url, headers=headers, params=params)
    html = response.text

    soup = BeautifulSoup(html, "html.parser")

    paragraphs = []
    titles = []

    # Iterate through all <p> tags
    for p in soup.find_all("p"):
        # Search for the nearest preceding heading tag within <h1> to <h5> range
        headings = p.find_previous_siblings(["h1", "h2", "h3", "h4", "h5"])
        title_found = False
        for heading in headings:
            if heading.find_next_sibling("p") == p:
                title = heading.text.strip()
                titles.append(title)
                title_found = True
                break  # Stop searching once a valid heading is found

        if not title_found:
            titles.append("no_title")

        paragraphs.append(p.text)

    return paragraphs, titles


def paragraph_splitter(paragraphs, titles):
    passages = []
    passage_titles = []

    for i, paragraph in enumerate(paragraphs):
        title = titles[i]

        if len(paragraph) == 0:
            continue

        words = paragraph.split()

        for j in range(0, len(words), 100):
            passage = " ".join(words[j : j + 100]).strip()

            if len(passage) == 0:
                continue

            passage_titles.append(title)
            passages.append(passage)

            chunked_corpus = {"title": passage_titles, "text": passages}
    return chunked_corpus, passage_titles, passages


def tokenize_paragraph(chunked_corpus):
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/facebook-dpr-ctx_encoder-multiset-base"
    )
    num_passages = len(chunked_corpus["title"])

    # print("Tokenizing {:,} passages for DPR...".format(num_passages))

    # Tokenize the whole dataset! This will take ~15 to 20 seconds.
    outputs = tokenizer(
        chunked_corpus["title"],
        chunked_corpus["text"],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    # print(" DONE.")

    # “input_ids’ holds the encoded tokens for the entire corpus.
    input_ids = outputs["input_ids"]

    return input_ids


def set_device():
    # If there's a GPU available.
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        # print("There are %d GPU(s) available." % torch.cuda.device_count())
        # print("We will use the GPU:", torch.cuda.get_device_name(0))

    # If not...
    else:
        # print("No GPU available! ")
        # You could use the CPU, but I don't recommend it!
        device = torch.device("cpu")
    ctx_encoder = DPRContextEncoder.from_pretrained(
        "facebook/dpr-ctx_encoder-multiset-base"
    )
    ctx_encoder = ctx_encoder.to(device=device)

    return ctx_encoder, device


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.

    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_embeddings(input_ids, ctx_encoder, device):
    # Disable gradients
    torch.set_grad_enabled(False)

    # Parameters
    batch_size = 16

    # Total passages
    num_passages = input_ids.size()[0]

    # Total batches
    num_batches = math.ceil(num_passages / batch_size)

    # Store embeddings here
    embeddings = []

    # print(f"Generating embeddings for {num_passages} passages...")

    for i in range(0, num_passages, batch_size):
        # Get batch
        batch_ids = input_ids[i : i + batch_size].to(device)

        # Get embeddings for batch
        batch_embeds = ctx_encoder(batch_ids)["pooler_output"].detach().cpu().numpy()

        # Append to list
        embeddings.append(batch_embeds)

    # Concatenate all
    embeddings = np.concatenate(embeddings, axis=0)

    # print(f"Generated {embeddings.shape[0]} embeddings")

    return embeddings


def add_faiss_index( embeddings):
    # "The dimension of the embeddings to pass to the HNSW Faiss index.”
    dim = 768

    # "The number of bi-directional links created for every new element during the
    # HNSW index construction.”
    m = 128

    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    # print("Building the FAISS index...")

    # Track elapsed time for progress updates.
    t0 = time.time()

    index.train(embeddings)

    index.add(embeddings)

    # print(" DONE.")

    # print(" Adding embeddings to index took", self.format_time(time.time() - t0))

    return index, dim, m


def generate_dataset(chunked_corpus, embeddings, index, dim, m, titles, paragraphs):
    # Create a DataFrame from the dictionary.
    chunked_corpus = {"title": titles, "text": paragraphs}
    df = pd.DataFrame(chunked_corpus)

    # Convert the DataFrame into a huggingface Dataset object.
    dataset = Dataset.from_pandas(df)
    embs = []

    # For each ehbedding...
    for i in range(embeddings.shape[0]):
        # Add it to the list
        embs.append(embeddings[i, :])
    dataset = dataset.add_column("embeddings", embs)
    # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)

    dataset.add_faiss_index(
        column="embeddings",
        index_name="embeddings",
        custom_index=index,
        faiss_verbose=True,
    )

    return dataset
