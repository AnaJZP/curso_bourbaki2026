"""
embeddings.py — Extracción de embeddings con CLIP (ViT-B/32).
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL_NAME, DEVICE


def load_clip():
    """Carga el modelo y procesador CLIP."""
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    return model, processor


def extract_embeddings(
    df: pd.DataFrame,
    processor: CLIPProcessor,
    model: CLIPModel,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae embeddings de texto e imagen para cada fila del DataFrame.

    Returns
    -------
    text_emb : ndarray de forma (N, 512)
    image_emb : ndarray de forma (N, 512)
    """
    text_embs, image_embs = [], []

    for start in tqdm(range(0, len(df), batch_size), desc="Extrayendo embeddings"):
        batch = df.iloc[start : start + batch_size]
        texts = batch["text"].astype(str).tolist()
        images = [Image.open(p).convert("RGB") for p in batch["img_path"]]

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        text_embs.append(outputs.text_embeds.cpu().numpy())
        image_embs.append(outputs.image_embeds.cpu().numpy())

    return np.vstack(text_embs), np.vstack(image_embs)
