# Indexation des données

Ce module prépare les données utilisées par **IA Search**.

Le script `main.py` :

1. télécharge le jeu de données ;
2. calcule les embeddings avec `qwen3-embedding:0.6b` via Ollama ;
3. regroupe les documents en 10 clusters avec KMeans ;
4. indexe les documents dans Elasticsearch (`data_1` à `data_10`) ;
5. indexe les centres des clusters dans `cluster_centers`.

## Exécution

Elasticsearch et Ollama doivent être accessibles sur `localhost`.

```bash
uv sync
uv run python main.py
```

Voir le `README.md` à la racine du dépôt pour l'installation complète.
