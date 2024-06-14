import requests
# from dotenv import load_dotenv
# load_dotenv()

def batchify(data, batch_size):
    """Divide the data into batches."""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def get_embeddings_for_batch(batch, hf_token, model):
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    response = requests.post(api_url, headers=headers, json={"inputs": batch, "options": {"wait_for_model": True}})
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    embeddings = response.json()
    return embeddings

def get_embeddings(sentences, hf_token, model):
    

    batches = batchify(sentences, 64)
    print(f'collected {len(batches)} batches')

    all_embeddings = []

    # Process each batch
    for i, batch in enumerate(batches):
        print(f'getting embeddings for batch {i}')
        embeddings = get_embeddings_for_batch(batch, hf_token, model)
        all_embeddings.extend(embeddings)
        print('ok')
    
    return all_embeddings