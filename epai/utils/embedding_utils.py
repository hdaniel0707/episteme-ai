import torch
import pandas as pd
from tqdm import tqdm


def add_embedding(df: pd.DataFrame, emb_type: str) -> pd.DataFrame:
    """Add an emb_{emb_type} column to df using title + sep + abstract as input.

    Supported emb_type values: specter2, scibert, openai.
    specter2/scibert require: transformers (+ adapters for specter2).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    col = f"emb_{emb_type}"

    if emb_type == "specter2":
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter(
            "allenai/specter2", source="hf", load_as="specter2", set_active=True
        )
        model.to(device)
        model.eval()

        def embed(title, abstract):
            text = title + tokenizer.sep_token + abstract
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            return out.last_hidden_state[:, 0, :].squeeze().cpu().tolist()

    elif emb_type == "scibert":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
        model.eval()

        def embed(title, abstract):
            text = title + tokenizer.sep_token + abstract
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                out = model(**inputs)
            return out.last_hidden_state[:, 0, :].squeeze().cpu().tolist()

    elif emb_type == "openai":
        from openai import OpenAI

        client = OpenAI()
        openai_model = "text-embedding-3-small"

        def embed(title, abstract):
            text = title + " [SEP] " + abstract
            response = client.embeddings.create(input=text, model=openai_model)
            return response.data[0].embedding

    else:
        raise ValueError(
            f"Unknown emb_type: {emb_type!r}. Choose from: specter2, scibert, openai"
        )

    embeddings = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding ({emb_type})"):
        embeddings.append(embed(row["title"] or "", row["abstract"] or ""))

    df[col] = embeddings
    return df
