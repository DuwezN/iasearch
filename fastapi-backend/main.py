from typing import Any
import asyncio
import os

import requests
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware

from rerank import rerank


app = FastAPI(title="iasearch")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


ELASTICSEARCH_HOST = os.getenv(
    "ELASTICSEARCH_HOST",
    "http://localhost:9200"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_URL = f"{OLLAMA_HOST}/api/embeddings"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CLUSTERS_INDEX = "cluster_centers"

# Seuil de pertinence : on ne garde que les documents avec rerank_score >= 0.4
RERANK_THRESHOLD = 0.4

# Requête hors sujet (aucun document au-dessus du seuil) :
# on renvoie quand même les meilleurs résultats du rerank
FALLBACK_SIZE = 3
FALLBACK_MESSAGE = "Cette recherche ne correspond pas suffisamment aux données disponibles."

es = Elasticsearch(ELASTICSEARCH_HOST)

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embedding"]


def apply_relevance_threshold(reranked_hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Seuil de pertinence, avec repli sur les meilleurs résultats si aucun ne le passe.

    reranked_hits est déjà trié par rerank_score décroissant.
    """
    relevant_hits = [
        hit for hit in reranked_hits
        if hit["rerank_score"] >= RERANK_THRESHOLD
    ]

    if relevant_hits:
        return {
            "fallback": False,
            "message": None,
            "results": relevant_hits,
        }

    return {
        "fallback": True,
        "message": FALLBACK_MESSAGE,
        "results": reranked_hits[:FALLBACK_SIZE],
    }


@app.get("/")
def root():
    return {"message": "IA Search API"}


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    size: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:

    # Les appels synchrones (requests, client Elasticsearch) sont exécutés
    # dans un thread pour ne pas bloquer la boucle asynchrone de FastAPI

    # 1. Transformer la recherche utilisateur en embedding
    query_embedding = await asyncio.to_thread(get_embedding, q)

    # 2. KNN : trouver le cluster le plus proche
    clusters = await asyncio.to_thread(
        es.search,
        index=CLUSTERS_INDEX,
        knn={
            "field": "center",
            "query_vector": query_embedding,
            "k": 1,
            "num_candidates": 10,
        },
    )

    cluster_hits = clusters["hits"]["hits"]

    if not cluster_hits:
        raise HTTPException(
            status_code=404,
            detail="No matching cluster",
        )

    # 3. Récupérer l'id du meilleur cluster
    cluster = cluster_hits[0]

    cluster_id = cluster["_source"]["cluster_id"]

    # Nos clusters sont 0-9
    # Nos index Elasticsearch sont data_1-data_10
    data_index = f"data_{cluster_id + 1}"

    # 4. Recherche hybride dans le cluster sélectionné :
    # BM25 sur line + KNN sur embedding, scores additionnés par Elasticsearch
    results = await asyncio.to_thread(
        es.search,
        index=data_index,
        query={
            "match": {
                "line": q
            }
        },
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": 100,
        },
        size=size,
    )

    hits = results["hits"]["hits"]

    # 5. Rerank avec le LLM : ajoute rerank_score (le _score ES est conservé)
    # et trie par rerank_score décroissant
    reranked_hits = await rerank(q, hits)

    # 6. Seuil de pertinence (repli sur les 3 meilleurs si hors sujet)
    return {
        "query": q,
        "cluster_id": cluster_id,
        "data_index": data_index,
        **apply_relevance_threshold(reranked_hits),
    }


@app.get("/search/{cluster_id}")
async def search_in_cluster(
    cluster_id: int = Path(..., ge=0, le=9),
    q: str = Query(..., min_length=1),
    size: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:
    """Recherche directement dans un cluster donné, sans KNN sur cluster_centers."""

    # 1. Transformer la recherche utilisateur en embedding
    query_embedding = await asyncio.to_thread(get_embedding, q)

    # 2. Cluster imposé : clusters 0-9 -> index data_1-data_10
    data_index = f"data_{cluster_id + 1}"

    # 3. Même recherche hybride que /search : BM25 sur line + KNN sur embedding
    results = await asyncio.to_thread(
        es.search,
        index=data_index,
        query={
            "match": {
                "line": q
            }
        },
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": 100,
        },
        size=size,
    )

    hits = results["hits"]["hits"]

    # 4. Même rerank que /search
    reranked_hits = await rerank(q, hits)

    # 5. Même seuil de pertinence que /search (repli sur les 3 meilleurs si hors sujet)
    return {
        "query": q,
        "cluster_id": cluster_id,
        "data_index": data_index,
        **apply_relevance_threshold(reranked_hits),
    }