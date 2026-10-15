from typing import Any
import asyncio
import json
import logging
import math
import os

import httpx


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"

RERANK_MODEL = "qwen3.5:0.8b"

OLLAMA_TIMEOUT = 120

logger = logging.getLogger(__name__)

# Sortie contrainte : {"answer": "yes"} ou {"answer": "no"}
ANSWER_FORMAT = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["yes", "no"],
        }
    },
    "required": ["answer"],
}

# Début de réponse imposé au modèle : le premier token généré
# est donc directement la valeur de "answer" (yes / no)
ANSWER_PREFIX = '{"answer": "'

SYSTEM_PROMPT = (
    "You are a strict relevance judge for a search engine. "
    "Answer yes only if the document directly and specifically answers or matches the user's query. "
    "Weak, indirect, broad, thematic or merely related topics, or documents that only share some words "
    "with the query, must be answered no. "
    # Exemples volontairement sans lien avec les requêtes de test
    "Example: query 'school bus transport', document 'Supply of office furniture' -> no. "
    "Example: query 'school bus transport', document 'Purchase of fuel for municipal vehicles' -> no. "
    "Example: query 'road maintenance', document 'Resurfacing works on municipal roads' -> yes. "
    'Answer only in JSON: {"answer": "yes"} or {"answer": "no"}.'
)


YES_TOKENS = {"yes", "Yes", "YES", "true", "True", "TRUE", "1"}
NO_TOKENS = {"no", "No", "NO", "false", "False", "FALSE", "0"}


def _normalize(token: str) -> str:
    # ' yes', '"yes' -> 'yes' (la casse est gérée par les listes de variantes)
    return token.strip().strip('"')


def _is_yes_no(token: str) -> bool:
    return _normalize(token) in YES_TOKENS | NO_TOKENS


def yes_no_probabilities(logprobs: list[dict[str, Any]]) -> tuple[float, float]:
    """Somme des probabilités des variantes yes (p_yes) et no (p_no)."""
    p_yes = 0.0
    p_no = 0.0

    for entry in logprobs:
        top_logprobs = entry.get("top_logprobs", [])

        # Position de la réponse : le token généré ou ses alternatives
        # contiennent une variante de yes / no
        if not (
            _is_yes_no(entry["token"])
            or any(_is_yes_no(t["token"]) for t in top_logprobs)
        ):
            continue

        candidates = list(top_logprobs)

        # Token réellement généré, s'il n'est pas déjà dans top_logprobs
        # (sinon sa probabilité serait comptée deux fois)
        if all(t["token"] != entry["token"] for t in top_logprobs):
            candidates.append(entry)

        for candidate in candidates:
            token = _normalize(candidate["token"])
            probability = math.exp(candidate["logprob"])

            if token in YES_TOKENS:
                p_yes += probability
            elif token in NO_TOKENS:
                p_no += probability

        # Une seule position de réponse
        break

    return p_yes, p_no


async def relevance_score(
    client: httpx.AsyncClient,
    query: str,
    document: str,
) -> float:
    """Probabilité que le document soit pertinent : p_yes / (p_yes + p_no)."""
    response = await client.post(
        OLLAMA_CHAT_URL,
        json={
            "model": RERANK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n"
                        f"Document: {document}\n"
                        "Does the document directly and specifically match the query?"
                    ),
                },
                # Indispensable : Ollama ne renvoie les logprobs que du premier
                # token généré, qui doit donc être la valeur de "answer"
                {"role": "assistant", "content": ANSWER_PREFIX},
            ],
            "format": ANSWER_FORMAT,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": 10,
            "options": {"temperature": 0},
        },
    )

    response.raise_for_status()

    data = response.json()

    # Somme des probabilités des variantes de yes et de no
    p_yes, p_no = yes_no_probabilities(data.get("logprobs") or [])

    if p_yes + p_no > 0:
        return p_yes / (p_yes + p_no)

    # Aucun token yes / no exploitable : on se rabat sur la réponse JSON
    content = data["message"]["content"]
    if not content.lstrip().startswith("{"):
        content = ANSWER_PREFIX + content

    try:
        answer = json.loads(content).get("answer")
    except json.JSONDecodeError:
        answer = None

    return 1.0 if answer == "yes" else 0.0


async def _score_document(
    client: httpx.AsyncClient,
    query: str,
    document: dict[str, Any],
) -> float:
    # Lecture de line ici pour qu'une erreur reste propre à ce document
    return await relevance_score(client, query, document["_source"]["line"])


async def rerank(query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ajoute rerank_score à chaque hit Elasticsearch et trie par score décroissant."""
    # Un seul client (connexions réutilisées), appels Ollama lancés en parallèle
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        scores = await asyncio.gather(
            *(_score_document(client, query, document) for document in documents),
            return_exceptions=True,
        )

    reranked = []

    for document, score in zip(documents, scores):
        # Un document en échec ne fait pas tomber le rerank : score 0, en fin de liste
        if isinstance(score, Exception):
            logger.warning(
                "Rerank failed for document %s: %r",
                document.get("_id"),
                score,
            )
            score = 0.0

        reranked.append({**document, "rerank_score": score})

    reranked.sort(key=lambda d: d["rerank_score"], reverse=True)

    return reranked
