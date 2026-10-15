import requests
import json
import numpy as np
from sklearn.cluster import KMeans


PCLOUD_API_URL = (
    "https://apilux3.pcloud.com/getpublinkdownload"
    "?fileid=93128819642"
    "&code=XZnqkRJZ98VUd3rEuMXAFIsEpAtnNHoKYK8k"
    "&linkpassword="
    "&forcedownload=1"
)

OLLAMA_API_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "qwen3-embedding:0.6b"

ELASTICSEARCH_URL = "http://localhost:9200"

NUM_CLUSTERS = 10
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32
ELASTICSEARCH_BULK_BATCH_SIZE = 100


def get_data_from_cloud():
    print("Récupération du fichier...")

    try:
        api_response = requests.get(
            PCLOUD_API_URL,
            timeout=30
        )
        api_response.raise_for_status()

        json_data = api_response.json()

        if (
            "path" in json_data
            and "hosts" in json_data
            and json_data["hosts"]
        ):
            relative_download_path = json_data["path"]
            host = json_data["hosts"][0]

            full_download_url = (
                f"https://{host}{relative_download_path}"
            )

            print(
                f"Direct download link found: "
                f"{full_download_url}"
            )

            file_content_response = requests.get(
                full_download_url,
                timeout=60
            )
            file_content_response.raise_for_status()

            lines = [
                line.strip()
                for line in file_content_response.text.splitlines()
                if line.strip()
            ]

            print(
                f"{len(lines)} lignes récupérées."
            )

            return lines

        print(
            "Error: 'path' or 'hosts' key not found "
            "in the JSON response, or hosts list is empty."
        )
        print(json.dumps(json_data, indent=2))
        return []

    except requests.exceptions.RequestException as e:
        print(
            f"Error making API request "
            f"or fetching file: {e}"
        )

    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON response: {e}"
        )

    except Exception as e:
        print(
            f"An unexpected error occurred: {e}"
        )

    return []


def get_data_embeddings(lines):
    print(
        "Calcul des embeddings avec Ollama..."
    )

    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        response.raise_for_status()

    except requests.exceptions.RequestException:
        print(
            "\nError: Ollama is not running "
            "or not accessible at "
            "http://localhost:11434"
        )
        return None

    embeddings = []

    for batch_start in range(
        0,
        len(lines),
        EMBEDDING_BATCH_SIZE
    ):
        batch = lines[
            batch_start:
            batch_start + EMBEDDING_BATCH_SIZE
        ]

        try:
            payload = {
                "model": MODEL_NAME,
                "input": batch
            }

            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            batch_embeddings = (
                response.json()["embeddings"]
            )

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Embedding count does not match batch size"
                )

            embeddings.extend(batch_embeddings)

        except requests.exceptions.RequestException as e:
            print(
                f"Error processing lines "
                f"{batch_start + 1}-"
                f"{batch_start + len(batch)}: {e}"
            )
            embeddings.extend(
                [[0.0] * EMBEDDING_DIM for _ in batch]
            )

        except (KeyError, TypeError, ValueError) as e:
            print(
                f"Error processing lines "
                f"{batch_start + 1}-"
                f"{batch_start + len(batch)}: {e}"
            )
            embeddings.extend(
                [[0.0] * EMBEDDING_DIM for _ in batch]
            )

        print(
            f"Processed "
            f"{batch_start + len(batch)}/{len(lines)} "
            f"lines..."
        )

    if not embeddings:
        print(
            "Aucun embedding calculé."
        )
        return None

    embeddings_array = np.array(
        embeddings,
        dtype=np.float32
    )

    print(
        f"Embeddings calculated. "
        f"Shape: "
        f"{embeddings_array.shape}"
    )

    return embeddings_array


def cluster_data(
    line_embeddings,
    lines
):
    print(
        f"\nClustering KMeans avec "
        f"{NUM_CLUSTERS} clusters..."
    )

    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=42,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(
        line_embeddings
    )
    labels = kmeans.labels_

    cluster_centers = (
        kmeans.cluster_centers_
    )

    cluster_to_lines = {
        i: []
        for i in range(NUM_CLUSTERS)
    }

    for i, cluster_id in enumerate(
        clusters
    ):
        cluster_to_lines[
            cluster_id
        ].append(lines[i])

    print(
        "\nRépartition des clusters :"
    )

    for cluster_id, cluster_lines in (
        cluster_to_lines.items()
    ):
        print(
            f"\nCluster {cluster_id}: "
            f"{len(cluster_lines)} lignes"
        )

        for line in cluster_lines[:3]:
            print(
                f" - {line}"
            )

    return (
        cluster_to_lines,
        cluster_centers,
        labels
    )


def index_clusters_in_elasticsearch(
    lines,
    line_embeddings,
    cluster_to_lines,
    labels
):
    print(
        "\nVérification de Elasticsearch..."
    )

    try:
        response = requests.get(
            f"{ELASTICSEARCH_URL}/_cat/health?h=status",
            timeout=5
        )
        response.raise_for_status()

        print(
            f"Elasticsearch OK : "
            f"{response.text.strip()}"
        )

    except requests.exceptions.RequestException as e:
        print(
            f"Elasticsearch inaccessible : {e}"
        )
        return

    print(
        f"\nIndexing {len(lines)} lines "
        f"into Elasticsearch..."
    )

    for cluster_id in cluster_to_lines:
        index_name = (
            f"data_{cluster_id + 1}"
        )

        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "line": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            }
        }

        try:
            response = requests.put(
                f"{ELASTICSEARCH_URL}/{index_name}",
                json=mapping,
                timeout=10
            )

            if response.status_code not in (
                200,
                400
            ):
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(
                f"Error creating index "
                f"{index_name}: {e}"
            )

    for batch_start in range(
        0,
        len(lines),
        ELASTICSEARCH_BULK_BATCH_SIZE
    ):
        batch_end = min(
            batch_start + ELASTICSEARCH_BULK_BATCH_SIZE,
            len(lines)
        )
        bulk_lines = []

        for i in range(batch_start, batch_end):
            cluster_id = int(labels[i])
            index_name = f"data_{cluster_id + 1}"
            doc = {
                "embedding": line_embeddings[i].tolist(),
                "line": lines[i]
            }

            bulk_lines.append(
                json.dumps({
                    "index": {
                        "_index": index_name,
                        "_id": i
                    }
                })
            )
            bulk_lines.append(json.dumps(doc))

        try:
            response = requests.post(
                f"{ELASTICSEARCH_URL}/_bulk",
                data="\n".join(bulk_lines) + "\n",
                headers={
                    "Content-Type": "application/x-ndjson"
                },
                timeout=60
            )
            response.raise_for_status()

            bulk_response = response.json()
            for offset, item in enumerate(
                bulk_response.get("items", [])
            ):
                index_result = item.get("index", {})
                if "error" in index_result:
                    print(
                        f"Error indexing line "
                        f"{batch_start + offset}: "
                        f"{index_result['error']}"
                    )

        except requests.exceptions.RequestException as e:
            print(
                f"Error indexing lines "
                f"{batch_start}-{batch_end - 1}: {e}"
            )
        except (ValueError, TypeError) as e:
            print(
                f"Invalid bulk response for lines "
                f"{batch_start}-{batch_end - 1}: {e}"
            )

        print(
            f"Indexed "
            f"{batch_end}/{len(lines)} lines..."
        )

    print(
        "\nFinished indexing lines."
    )

    for cluster_id in cluster_to_lines:
        index_name = (
            f"data_{cluster_id + 1}"
        )

        try:
            response = requests.get(
                f"{ELASTICSEARCH_URL}/"
                f"{index_name}/_count",
                timeout=5
            )

            response.raise_for_status()

            count = (
                response
                .json()
                .get("count", 0)
            )

            print(
                f"{index_name}: "
                f"{count} documents"
            )

        except requests.exceptions.RequestException:
            pass


def index_data_clusters(
    cluster_centers
):
    INDEX_NAME = "cluster_centers"

    print(
        f"\nIndexing "
        f"{len(cluster_centers)} "
        f"cluster centers..."
    )

    mapping = {
        "mappings": {
            "properties": {
                "center": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine"
                },
                "cluster_id": {
                    "type": "long"
                }
            }
        }
    }

    try:
        response = requests.put(
            f"{ELASTICSEARCH_URL}/"
            f"{INDEX_NAME}",
            json=mapping,
            timeout=10
        )

        if response.status_code not in (
            200,
            400
        ):
            response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(
            f"Error creating index "
            f"{INDEX_NAME}: {e}"
        )
        return

    for cluster_id, center in enumerate(
        cluster_centers
    ):
        doc = {
            "cluster_id": cluster_id,
            "center": (
                center.tolist()
            )
        }

        try:
            response = requests.post(
                f"{ELASTICSEARCH_URL}/"
                f"{INDEX_NAME}/_doc/"
                f"{cluster_id}",
                json=doc,
                timeout=10
            )

            response.raise_for_status()

            print(
                f"Indexed cluster center "
                f"{cluster_id}"
            )

        except requests.exceptions.RequestException as e:
            print(
                f"Error indexing cluster "
                f"center {cluster_id}: {e}"
            )

    print(
        f"\nFinished indexing "
        f"cluster centers into "
        f"{INDEX_NAME}."
    )


def main():
    lines = get_data_from_cloud()

    if not lines:
        print(
            "Aucune donnée récupérée."
        )
        return

    line_embeddings = (
        get_data_embeddings(
            lines
        )
    )

    if line_embeddings is None:
        print(
            "Impossible de continuer "
            "sans embeddings."
        )
        return

    (
        cluster_to_lines,
        cluster_centers,
        labels
    ) = cluster_data(
        line_embeddings,
        lines
    )

    index_clusters_in_elasticsearch(
        lines,
        line_embeddings,
        cluster_to_lines,
        labels
    )

    index_data_clusters(
        cluster_centers
    )

    print(
        "\nTraitement terminé."
    )

    print(
        f"Nombre de lignes : "
        f"{len(lines)}"
    )

    print(
        f"Nombre d'embeddings : "
        f"{len(line_embeddings)}"
    )

    print(
        f"Nombre de clusters : "
        f"{len(cluster_centers)}"
    )


if __name__ == "__main__":
    main()