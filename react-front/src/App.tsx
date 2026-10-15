import { useState, type FormEvent } from 'react'
import './App.css'

// Types de la réponse GET /search du backend FastAPI
interface SearchSource {
  line: string
}

interface SearchHit {
  _id: string
  _index: string
  _score: number // score Elasticsearch (BM25 + KNN)
  _source: SearchSource
  rerank_score: number // score final de pertinence (rerank LLM, >= 0.4)
}

interface SearchResponse {
  query: string
  cluster_id: number
  data_index: string
  // true : aucun résultat >= 0.4, results contient les 3 meilleurs du rerank
  fallback: boolean
  message: string | null
  results: SearchHit[]
}

const API_URL = '/api/search'
const RESULT_SIZE = 10

interface Relevance {
  label: string
  level: 'high' | 'medium' | 'low' | 'none'
}

// Sous 0.4 : uniquement en mode fallback (requête hors sujet)
function relevance(score: number): Relevance {
  if (score >= 0.8) return { label: 'Très pertinent', level: 'high' }
  if (score >= 0.6) return { label: 'Pertinent', level: 'medium' }
  if (score >= 0.4) return { label: 'Faiblement pertinent', level: 'low' }
  return { label: 'Non pertinent', level: 'none' }
}

async function searchApi(query: string): Promise<SearchResponse> {
  const params = new URLSearchParams({ q: query, size: String(RESULT_SIZE) })
  const response = await fetch(`${API_URL}?${params}`)

  if (!response.ok) {
    // FastAPI renvoie les erreurs sous la forme { "detail": ... }
    let detail = response.statusText
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      // réponse non JSON (ex. erreur Nginx) : on garde le statusText
    }
    throw new Error(`Erreur ${response.status} : ${detail}`)
  }

  return (await response.json()) as SearchResponse
}

function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const q = query.trim()
    if (!q) return

    setLoading(true)
    setError(null)
    setData(null)

    try {
      setData(await searchApi(q))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="search-page">
      <h1>IA Search</h1>

      <form className="search-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ex. travaux de rénovation"
          aria-label="Texte de recherche"
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? 'Recherche…' : 'Search'}
        </button>
      </form>

      {loading && <p className="status">Recherche en cours…</p>}

      {error && <p className="status error">{error}</p>}

      {data && (
        <section className="results">
          <dl className="meta">
            <div>
              <dt>Requête</dt>
              <dd>{data.query}</dd>
            </div>
            <div>
              <dt>Cluster</dt>
              <dd>{data.cluster_id}</dd>
            </div>
            <div>
              <dt>Index</dt>
              <dd>{data.data_index}</dd>
            </div>
            <div>
              <dt>Résultats</dt>
              <dd>{data.results.length}</dd>
            </div>
          </dl>

          {data.fallback && (
            <p className="fallback-notice" role="status">
              Cette recherche ne correspond pas suffisamment aux données
              disponibles. Voici néanmoins les résultats les plus proches :
            </p>
          )}

          {data.results.length === 0 ? (
            <p className="status">
              Aucun résultat pertinent trouvé pour cette requête.
            </p>
          ) : (
            <ol className="result-list">
              {data.results.map((hit) => {
                const { label, level } = relevance(hit.rerank_score)
                return (
                  <li key={hit._id}>
                    <div className="scores">
                      <span className="score" title="Score de pertinence (rerank)">
                        {hit.rerank_score.toFixed(3)}
                      </span>
                      <span className={`relevance ${level}`}>{label}</span>
                    </div>
                    <div className="result-body">
                      <p>{hit._source.line}</p>
                      <span className="es-score">
                        Score Elasticsearch : {hit._score.toFixed(3)}
                      </span>
                    </div>
                  </li>
                )
              })}
            </ol>
          )}
        </section>
      )}
    </main>
  )
}

export default App
