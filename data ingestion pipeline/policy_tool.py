"""Search the ingested policy manuals through a LangChain tool.

The tool embeds the question with the same OpenAI model used at ingestion,
then queries the persistent Chroma collection. Results include the paragraph,
its topic, and the original policy file so an agent can cite the source.

    export OPENAI_API_KEY="your-key"
    python "data ingestion pipeline/ingest.py"
    python "data ingestion pipeline/policy_tool.py" "when must a claim be acknowledged"
"""

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import chromadb
from langchain.tools import tool
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "DB"
DEFAULT_RESULTS = 4
_STOP = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "is", "are",
    "be", "by", "with", "that", "this", "how", "what", "when", "who", "must",
    "may", "do", "does", "did", "not", "from", "over", "its", "their", "they",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP and len(token) > 2
    }


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 5:
            return token[: -len(suffix)]
    return token


def _as_passage(chunk: dict) -> dict:
    return {
        "doc_id": chunk.get("doc_id", ""),
        "title": chunk.get("title", ""),
        "topic": chunk.get("topic", ""),
        "reference": chunk.get("reference", ""),
        "text": chunk.get("text", ""),
        "distance": 1.0,
    }


def _supplements(query: str, passages: list[dict]) -> list[dict]:
    """Add the paragraph that shares the question's words, and a matching definition.

    The four nearest embeddings often land in the right manual and miss the
    sentence that states the rule. These two lookups read the chunk file only.
    """
    path = DB_DIR / "chunks.json"
    if not path.exists():
        return []
    chunks = json.loads(path.read_text(encoding="utf-8"))
    question = _tokens(query)
    question_stems = {_stem(token) for token in question}
    seen = {passage["text"] for passage in passages}
    extras = []

    def lexical(chunk: dict) -> int:
        return len(question & _tokens(chunk["text"]))

    ranked = sorted(chunks, key=lexical, reverse=True)
    for chunk in ranked:
        if lexical(chunk) < 3 or chunk["text"] in seen:
            continue
        extras.append(_as_passage(chunk))
        seen.add(chunk["text"])
        if len(extras) == 2:
            break

    def topic_key(chunk: dict) -> tuple:
        raw = [
            token
            for token in re.findall(r"[a-z0-9]+", chunk["topic"].lower())
            if token not in _STOP and len(token) > 2
        ]
        topic = [_stem(token) for token in raw]
        hits = len(set(topic) & question_stems)
        first = 1 if topic and topic[0] in question_stems else 0
        defined = 1 if chunk["text"].startswith("**") else 0
        coverage = hits / len(topic) if topic else 0
        return (first, defined, hits, coverage, -len(chunk["text"]))

    topic_added = 0
    for chunk in sorted(chunks, key=topic_key, reverse=True):
        if topic_key(chunk)[2] < 1 or chunk["text"] in seen:
            continue
        extras.append(_as_passage(chunk))
        seen.add(chunk["text"])
        topic_added += 1
        if topic_added == 2:
            break
    return extras


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
    passages.extend(_supplements(query, passages))
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
