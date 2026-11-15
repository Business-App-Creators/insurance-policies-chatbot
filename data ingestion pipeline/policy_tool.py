"""Search the ingested policy manuals through a LangChain tool.

The tool embeds the question with the same OpenAI model used at ingestion,
then queries the persistent Chroma collection. Results include the paragraph,
its topic, and the original policy file so an agent can cite the source.

    export OPENAI_API_KEY="your-key"
    python "data ingestion pipeline/ingest.py"
    python "data ingestion pipeline/policy_tool.py" "when must a claim be acknowledged"
"""

import importlib.util
import os
import sys
from pathlib import Path

import chromadb
from langchain.tools import tool
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "DB"
DEFAULT_RESULTS = 4


def _store():
    path = DB_DIR / "embedding-chunks.py"
    spec = importlib.util.spec_from_file_location("embedding_chunks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collection():
    store = _store()
    if not store.CHROMA_PATH.exists():
        raise RuntimeError(
            "No policy database at "
            f"{store.CHROMA_PATH}. Run the ingestion pipeline first."
        )
    client = chromadb.PersistentClient(path=str(store.CHROMA_PATH))
    names = [collection.name for collection in client.list_collections()]
    if store.COLLECTION_NAME not in names:
        raise RuntimeError(
            f"Collection '{store.COLLECTION_NAME}' is missing. "
            "Run the ingestion pipeline first."
        )
    return client.get_collection(store.COLLECTION_NAME), store


def search_passages(query: str, k: int = DEFAULT_RESULTS) -> list[dict]:
    """Return the closest policy passages as records the graph can cite."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before searching the policy manuals.")
    collection, store = _collection()
    embedded = OpenAI().embeddings.create(model=store.EMBEDDING_MODEL, input=query)
    found = collection.query(
        query_embeddings=[embedded.data[0].embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = (found.get("documents") or [[]])[0]
    metadatas = (found.get("metadatas") or [[]])[0]
    distances = (found.get("distances") or [[]])[0]
    passages = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        passages.append(
            {
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
                "topic": meta.get("topic", ""),
                "reference": meta.get("reference", ""),
                "text": text,
                "distance": distance,
            }
        )
    return passages


def search_manuals(query: str, k: int = DEFAULT_RESULTS) -> str:
    """Return the closest policy passages for a question."""
    passages = search_passages(query, k)
    if not passages:
        return "No matching policy passage was found."
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {passage['doc_id']} — {passage['topic']}",
                    f"Reference: {passage['reference']}",
                    passage["text"],
                ]
            )
        )
    return "\n\n".join(blocks)


@tool
def search_policy_manuals(query: str) -> str:
    """Search the Northline Assurance claims policy manuals.

    Use this for questions about claim handling: notice, coverage, investigation,
    reserves, payment, denial, fraud referral, recovery, complaints, or catastrophe
    claims. Answer from the returned passages and name the reference. If nothing
    relevant is returned, say the manuals do not decide the question.

    Args:
        query: The claims question to look up in the policy library.
    """
    return search_manuals(query)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python "data ingestion pipeline/policy_tool.py" "your question"')
    print(search_policy_manuals.invoke({"query": question}))
