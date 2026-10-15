from typing import Any
import os

import requests
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


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


@app.get("/")
def root():
    return {"message": "IA Search API"}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    size: int = Query(10, ge=1, le=100),
) -> dict[str, Any]:

    # 1. Transformer la recherche utilisateur en embedding
    query_embedding = get_embedding(q)

    # 2. KNN : trouver le cluster le plus proche
    clusters = es.search(
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
    results = es.search(
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

    return {
        "query": q,
        "cluster_id": cluster_id,
        "data_index": data_index,
        "results": results["hits"]["hits"],
    }