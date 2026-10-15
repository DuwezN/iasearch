# FastAPI backend

Backend de **IA Search**.

Il orchestre :

- la génération de l'embedding de la requête avec `qwen3-embedding:0.6b` ;
- la sélection du cluster dans Elasticsearch ;
- la recherche hybride BM25 + KNN ;
- le reranking avec `qwen3.5:0.8b` ;
- le seuil de pertinence et le fallback sur les trois meilleurs candidats.

## Variables d'environnement

- `ELASTICSEARCH_HOST` — défaut : `http://localhost:9200`
- `OLLAMA_HOST` — défaut : `http://localhost:11434`

Dans Docker Compose, ces valeurs pointent vers les services `elasticsearch` et `ollama`.

## API

- `GET /search`
- `GET /search/{cluster_id}`
- Swagger : `GET /docs`

Pour l'utilisation complète du projet, voir le `README.md` à la racine du dépôt.
