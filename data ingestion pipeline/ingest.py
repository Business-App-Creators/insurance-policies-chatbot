"""Ingest policy manuals and keep the collection current.

The first run inserts every chunk. Later runs insert new chunks, update chunks
whose text changed, and delete chunks that are no longer in the manuals.
Unchanged chunks are left as they are, so they are not embedded again.

    export OPENAI_API_KEY="your-key"
    python "data ingestion pipeline/ingest.py"
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import chromadb
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "DB"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def content_hash(embedding, chunk: dict) -> str:
    payload = embedding.embedding_input(chunk)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def open_collection(embedding):
    embedding.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(embedding.CHROMA_PATH))
    return client.get_or_create_collection(
        name=embedding.COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": embedding.EMBEDDING_MODEL},
    )


def stored_hashes(collection) -> dict[str, str]:
    if collection.count() == 0:
        return {}
    stored = collection.get(include=["metadatas"])
    hashes = {}
    for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
        hashes[chunk_id] = (metadata or {}).get("content_hash", "")
    return hashes


def metadata_for(chunk: dict, digest: str) -> dict:
    return {
        "topic": chunk["topic"],
        "section": chunk["section"],
        "reference": chunk["reference"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "content_hash": digest,
    }


def write_chunks(embedding, collection, chunks: list[dict]) -> None:
    if not chunks:
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before embedding new or changed chunks.")
    client = OpenAI()
    vectors = embedding.embed_texts(client, [embedding.embedding_input(chunk) for chunk in chunks])
    for start in range(0, len(chunks), embedding.BATCH_SIZE):
        batch = chunks[start : start + embedding.BATCH_SIZE]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=vectors[start : start + embedding.BATCH_SIZE],
            metadatas=[metadata_for(chunk, chunk["content_hash"]) for chunk in batch],
        )


def delete_chunks(embedding, collection, chunk_ids: list[str]) -> None:
    for start in range(0, len(chunk_ids), embedding.BATCH_SIZE):
        collection.delete(ids=chunk_ids[start : start + embedding.BATCH_SIZE])


def sync(chunks: list[dict], embedding) -> dict[str, int]:
    """Insert new chunks, update changed ones, and delete chunks that were removed."""
    collection = open_collection(embedding)
    current = {}
    for chunk in chunks:
        digest = content_hash(embedding, chunk)
        current[chunk["id"]] = {**chunk, "content_hash": digest}

    previous = stored_hashes(collection)
    to_write = [
        chunk
        for chunk_id, chunk in current.items()
        if previous.get(chunk_id) != chunk["content_hash"]
    ]
    to_delete = [chunk_id for chunk_id in previous if chunk_id not in current]

    write_chunks(embedding, collection, to_write)
    delete_chunks(embedding, collection, to_delete)

    inserted = sum(chunk["id"] not in previous for chunk in to_write)
    return {
        "inserted": inserted,
        "updated": len(to_write) - inserted,
        "deleted": len(to_delete),
        "unchanged": len(current) - len(to_write),
        "total": collection.count(),
    }


def main() -> None:
    chunking = load_module("chunking_docs", DB_DIR / "chunking-docs.py")
    embedding = load_module("embedding_chunks", DB_DIR / "embedding-chunks.py")
    chunks = chunking.chunk_policies()
    chunking.CHUNKS_PATH.write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = sync(chunks, embedding)
    print(
        "Ingestion complete: "
        f"{counts['inserted']} inserted, {counts['updated']} updated, "
        f"{counts['deleted']} deleted, {counts['unchanged']} unchanged. "
        f"{counts['total']} chunks in {embedding.CHROMA_PATH}."
    )
    print("The LangChain tool is search_policy_manuals.")


if __name__ == "__main__":
    main()
