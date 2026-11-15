"""OpenAI embedding helpers for the policy chunks.

Insertion and updates are handled by the data ingestion pipeline. This module
only embeds text and names the persistent collection the pipeline writes to.

    export OPENAI_API_KEY="your-key"
    python "data ingestion pipeline/ingest.py"
"""

import importlib.util
from pathlib import Path

from openai import OpenAI

DB_DIR = Path(__file__).resolve().parent
CHROMA_PATH = DB_DIR / "chroma"
COLLECTION_NAME = "policy_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Return one embedding per text, in the same order."""
    vectors: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in response.data:
            vectors[start + item.index] = item.embedding
    return vectors


def embedding_input(chunk: dict) -> str:
    """Text sent to the embedder. Topic and source stay attached to the paragraph."""
    return f"Topic: {chunk['topic']}\nReference: {chunk['reference']}\n\n{chunk['text']}"


def main() -> None:
    pipeline = Path(__file__).resolve().parents[1] / "data ingestion pipeline" / "ingest.py"
    spec = importlib.util.spec_from_file_location("policy_ingest", pipeline)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {pipeline}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
