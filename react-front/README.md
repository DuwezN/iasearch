# Frontend React

Interface web de **IA Search**, développée avec React, TypeScript et Vite.

En production Docker, Nginx sert les fichiers statiques et transmet les requêtes `/api/*` vers le backend FastAPI.

L'interface affiche :

- la requête, le cluster sélectionné et l'index Elasticsearch ;
- le `rerank_score` comme score principal ;
- le score Elasticsearch en information secondaire ;
- un niveau de pertinence et une barre visuelle ;
- un avertissement et les trois meilleurs candidats en mode fallback.

## Build

```bash
npm ci
npm run build
```

Pour lancer l'application complète, utiliser Docker Compose depuis la racine du dépôt.
