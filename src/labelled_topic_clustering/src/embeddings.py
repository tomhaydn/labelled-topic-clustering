from typing import List, Any
import requests
from requests.exceptions import HTTPError

def batchify(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Divide the data into batches."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def get_embeddings_for_batch(
    batch: List[str],
    hf_token: str,
    model: str
) -> List[List[float]]:
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": batch, "options": {"wait_for_model": True}},
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPError(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

    embeddings = response.json()
    return embeddings

def get_embeddings(
    sentences: List[str],
    hf_token: str,
    model: str,
    debug: bool
) -> List[List[float]]:
    batches = batchify(sentences, 64)

    if debug:
        print(f'collected {len(batches)} batches')

    all_embeddings = []

    # Process each batch
    for i, batch in enumerate(batches):
        if debug:
            print(f'getting embeddings for batch {i}')

        embeddings = get_embeddings_for_batch(batch, hf_token, model)
        all_embeddings.extend(embeddings)

        if debug:
            print('ok')

    return all_embeddings
