import json
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
ELASTICSEARCH_URL = "http://localhost:9200"
BATCH_SIZE = 32


def get_data_embeddings(
    data: str | Sequence[str],
    *,
    model: str = EMBEDDING_MODEL,
    ollama_url: str = OLLAMA_URL,
) -> list[list[float]]:
    """Calcule les embeddings d'un texte ou d'une liste de textes avec Ollama."""
    if isinstance(data, str):
        inputs = [data]
    else:
        inputs = list(data)

    if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
        raise ValueError("data doit contenir au moins un texte non vide")

    body = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    request = Request(
        f"{ollama_url.rstrip('/')}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama a retourné HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Impossible de joindre Ollama sur {ollama_url}") from error

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
        raise RuntimeError("Réponse Ollama invalide: champ embeddings absent ou incomplet")

    return embeddings


def _elasticsearch_request(
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> dict:
    request = Request(
        f"{ELASTICSEARCH_URL}{path}",
        data=body,
        headers={"Content-Type": content_type},
        method=method,
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response) if response.length != 0 else {}
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Elasticsearch a retourne HTTP {error.code} pour {path}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Impossible de joindre Elasticsearch sur {ELASTICSEARCH_URL}"
        ) from error


def _ensure_index(
    index_name: str,
    dimensions: int,
    *,
    metadata_field: str = "line",
    metadata_type: str = "text",
) -> None:
    request = Request(f"{ELASTICSEARCH_URL}/{index_name}", method="HEAD")
    try:
        with urlopen(request, timeout=30):
            return
    except HTTPError as error:
        if error.code != 404:
            raise RuntimeError(
                f"Impossible de verifier l'index {index_name}: HTTP {error.code}"
            ) from error
    except URLError as error:
        raise RuntimeError(
            f"Impossible de joindre Elasticsearch sur {ELASTICSEARCH_URL}"
        ) from error

    mapping = {
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimensions,
                    "index": True,
                    "similarity": "cosine",
                },
                metadata_field: {"type": metadata_type},
            }
        }
    }
    _elasticsearch_request(
        "PUT",
        f"/{index_name}",
        body=json.dumps(mapping).encode("utf-8"),
    )


def _index_batch(
    index_name: str, lines: Sequence[str], embeddings: Sequence[Sequence[float]]
) -> None:
    operations: list[str] = []
    for line, embedding in zip(lines, embeddings, strict=True):
        document_id = hashlib.sha256(line.encode("utf-8")).hexdigest()
        operations.append(
            json.dumps({"index": {"_index": index_name, "_id": document_id}})
        )
        operations.append(json.dumps({"embedding": embedding, "line": line}))

    body = ("\n".join(operations) + "\n").encode("utf-8")
    result = _elasticsearch_request(
        "POST",
        "/_bulk?refresh=true",
        body=body,
        content_type="application/x-ndjson",
    )
    if result.get("errors"):
        failures = [
            item["index"].get("error")
            for item in result.get("items", [])
            if item["index"].get("error")
        ]
        raise RuntimeError(f"Echec de l'indexation dans {index_name}: {failures}")


def _index_cluster(data_index: str, embedding: Sequence[float]) -> None:
    _ensure_index(
        "data_clusters",
        len(embedding),
        metadata_field="data_index",
        metadata_type="keyword",
    )
    document = {"embedding": embedding, "data_index": data_index}
    _elasticsearch_request(
        "PUT",
        f"/data_clusters/_doc/{data_index}?refresh=true",
        body=json.dumps(document).encode("utf-8"),
    )


def main(clusters: Sequence[Sequence[str]] | None = None) -> None:
    """Calcule et indexe les embeddings, avec un index Elasticsearch par cluster."""
    if clusters is None:
        if len(sys.argv) < 2:
            raise SystemExit(
                "Usage: python embeddings.py <cluster_1.txt> <cluster_2.txt> ..."
            )
        clusters = [
            [line for line in Path(filename).read_text(encoding="utf-8").splitlines() if line]
            for filename in sys.argv[1:]
        ]

    for cluster_number, cluster in enumerate(clusters, start=1):
        lines = [line for line in cluster if line.strip()]
        if not lines:
            continue

        index_name = f"data_{cluster_number}"
        indexed_count = 0
        embedding_sum: list[float] | None = None
        for start in range(0, len(lines), BATCH_SIZE):
            batch = lines[start : start + BATCH_SIZE]
            embeddings = get_data_embeddings(batch)
            _ensure_index(index_name, len(embeddings[0]))
            _index_batch(index_name, batch, embeddings)

            if embedding_sum is None:
                embedding_sum = [0.0] * len(embeddings[0])
            for embedding in embeddings:
                for dimension, value in enumerate(embedding):
                    embedding_sum[dimension] += value
            indexed_count += len(batch)

        if embedding_sum is None:
            continue
        cluster_embedding = [value / indexed_count for value in embedding_sum]
        _index_cluster(index_name, cluster_embedding)
        print(f"{index_name}: {indexed_count} lignes indexees")


if __name__ == "__main__":
    main()
