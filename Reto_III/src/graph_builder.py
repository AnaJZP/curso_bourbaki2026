"""
graph_builder.py -- Construccion de grafos para el Reto III.

Dos estrategias:
1. Grafo geografico: KNN basado en distancia haversine (lat/lon)
2. Grafo textual: similitud coseno entre embeddings de texto
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import BallTree

from .config import (
    COSINE_THRESHOLD,
    DEVICE,
    K_NEIGHBORS,
    SEED,
    TEXT_EMBED_MODEL,
)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Distancia haversine en km entre dos puntos."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def build_geo_graph(df, X, y, k=K_NEIGHBORS):
    """
    Construye un grafo KNN basado en geolocalizacion.

    Cada nodo se conecta con sus K vecinos mas cercanos por distancia haversine.

    Retorna: torch_geometric.data.Data
    """
    from torch_geometric.data import Data

    coords = np.radians(df[["latitude", "longitude"]].values)
    tree = BallTree(coords, metric="haversine")
    distances, indices = tree.query(coords, k=k + 1)  # +1 porque incluye a si mismo

    src_list = []
    dst_list = []
    for i in range(len(df)):
        for j_idx in range(1, k + 1):  # saltar indice 0 (si mismo)
            neighbor = indices[i, j_idx]
            src_list.append(i)
            dst_list.append(neighbor)
            # Agregar arista inversa (grafo no dirigido)
            src_list.append(neighbor)
            dst_list.append(i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Eliminar aristas duplicadas
    edge_index = torch.unique(edge_index, dim=1)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
        edge_index=edge_index,
    )

    print(f"  Grafo geografico: {data.num_nodes} nodos, "
          f"{data.num_edges} aristas, K={k}")

    return data


def build_text_graph(texts, X, y, threshold=COSINE_THRESHOLD):
    """
    Construye un grafo basado en similitud coseno de embeddings de texto.

    Intenta usar sentence-transformers; si no esta disponible, usa TF-IDF + SVD.

    Retorna: torch_geometric.data.Data
    """
    from torch_geometric.data import Data

    embeddings = _compute_text_embeddings(texts)

    # Calcular similitud coseno por bloques (para no explotar memoria)
    n = len(texts)
    src_list = []
    dst_list = []
    block_size = 500

    for i_start in range(0, n, block_size):
        i_end = min(i_start + block_size, n)
        sim_block = cosine_similarity(embeddings[i_start:i_end], embeddings)

        for i_local in range(i_end - i_start):
            i_global = i_start + i_local
            for j in range(n):
                if i_global != j and sim_block[i_local, j] > threshold:
                    src_list.append(i_global)
                    dst_list.append(j)

    if len(src_list) == 0:
        # Si el umbral es muy alto, bajar al top-K por similitud
        print(f"  Advertencia: umbral {threshold} no genero aristas. "
              "Usando top-{K_NEIGHBORS} por similitud.")
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, -1)
        for i in range(n):
            top_k = np.argsort(sim_matrix[i])[-K_NEIGHBORS:]
            for j in top_k:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
        edge_index=edge_index,
    )

    print(f"  Grafo textual: {data.num_nodes} nodos, "
          f"{data.num_edges} aristas, umbral={threshold}")

    return data


def _compute_text_embeddings(texts):
    """Calcula embeddings de texto con sentence-transformers o fallback TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer

        print("  Generando embeddings con sentence-transformers ...")
        model = SentenceTransformer(TEXT_EMBED_MODEL)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        print(f"  Embeddings: {embeddings.shape}")
        return embeddings

    except ImportError:
        print("  sentence-transformers no disponible, usando TF-IDF + SVD ...")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        X_tfidf = tfidf.fit_transform(texts)
        svd = TruncatedSVD(n_components=128, random_state=SEED)
        embeddings = svd.fit_transform(X_tfidf)
        print(f"  Embeddings TF-IDF+SVD: {embeddings.shape}")
        return embeddings
