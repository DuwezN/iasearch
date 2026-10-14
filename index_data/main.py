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

OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "qwen3-embedding:0.6b"

NUM_CLUSTERS = 10


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
                for line
                in file_content_response.text.splitlines()
                if line.strip()
            ]

            print(
                f"{len(lines)} lignes récupérées."
            )

            return lines

        else:
            print(
                "Error: 'path' or 'hosts' key not found "
                "in the JSON response, or hosts list is empty."
            )
            print(
                json.dumps(
                    json_data,
                    indent=2
                )
            )

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

    # Vérifie que Ollama répond
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

    for i, line in enumerate(lines):
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": line
            }

            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            embeddings.append(
                data["embedding"]
            )

            if (i + 1) % 10 == 0:
                print(
                    f"Processed "
                    f"{i + 1}/{len(lines)} "
                    f"lines..."
                )

        except requests.exceptions.RequestException as e:
            print(
                f"Error processing "
                f"line {i + 1}: {e}"
            )

        except KeyError:
            print(
                f"Error processing line "
                f"{i + 1}: "
                f"'embedding' not found "
                f"in Ollama response."
            )

    if not embeddings:
        print(
            "Aucun embedding calculé."
        )
        return None

    embeddings_array = np.array(
        embeddings
    )

    print(
        f"Embeddings calculated. "
        f"Shape: "
        f"{embeddings_array.shape}"
    )

    print(
        "\nFirst 5 embeddings "
        "(first 10 dimensions each):"
    )

    for i, embedding in enumerate(
        embeddings_array[:5]
    ):
        print(
            f"Line {i + 1}: "
            f"{embedding[:10].tolist()}..."
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

    cluster_centers = (
        kmeans.cluster_centers_
    )

    print(
        "\nRépartition des clusters :"
    )

    for cluster_id in range(
        NUM_CLUSTERS
    ):
        indices = np.where(
            clusters == cluster_id
        )[0]

        print(
            f"\nCluster {cluster_id}: "
            f"{len(indices)} lignes"
        )

        for index in indices[:3]:
            print(
                f" - {lines[index]}"
            )

    return (
        clusters,
        cluster_centers
    )


def main():
    lines = get_data_from_cloud()

    if not lines:
        print(
            "Aucune donnée récupérée."
        )
        return

    line_embeddings = (
        get_data_embeddings(lines)
    )

    if line_embeddings is None:
        print(
            "Impossible de continuer "
            "sans embeddings."
        )
        return

    clusters, cluster_centers = (
        cluster_data(
            line_embeddings,
            lines
        )
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

    return (
        lines,
        line_embeddings,
        clusters,
        cluster_centers
    )


if __name__ == "__main__":
    main()