"""Sequential cited chat: rewrite, retrieve, generate, verify, refuse.

Generate calls the chat model. The printed reply is the model's own words,
then the quoted passage from Chroma, then the citation. Verify checks that
the quote was copied from a retrieved passage. The model does not choose
the next node.

    python -m chatbot.agents.sequential.graph "When is a denial sent?"
"""

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

ROOT = Path(__file__).resolve().parents[3]
POLICY_TOOL = ROOT / "data ingestion pipeline" / "policy_tool.py"
REFUSAL = "The files do not contain that."
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
MAX_ATTEMPTS = 5
# rewrite, retrieve, then generate and verify up to five times, then refuse.
RECURSION_LIMIT = 2 + (2 * MAX_ATTEMPTS) + 2


class Citation(TypedDict, total=False):
    doc_id: str
    quote: str
    topic: str
    reference: str


class Passage(TypedDict):
    doc_id: str
    title: str
    topic: str
    reference: str
    text: str
    distance: float


class State(TypedDict, total=False):
    question: str
    history: list[dict]
    standalone_question: str
    retrieved: list[Passage]
    answer: str
    citations: list[Citation]
    citation_ok: bool
    attempts: int
    refusal: bool
    verify_note: str


def _policy_tool():
    spec = importlib.util.spec_from_file_location("policy_tool", POLICY_TOOL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {POLICY_TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def rewrite(state: State) -> dict:
    """Turn a follow-up into one standalone question. Do not answer it."""
    question = state["question"]
    history = state.get("history") or []
    if not history:
        return {"standalone_question": question, "attempts": 0}
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before asking the chat model.")
    transcript = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history
    )
    prompt = (
        "Rewrite the follow-up as one standalone question. "
        "Resolve words like 'that' from the conversation. Do not answer it.\n\n"
        f"Conversation:\n{transcript}\n\nFollow-up: {question}\n\nStandalone question:"
    )
    response = ChatOpenAI(model=CHAT_MODEL, temperature=0).invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {"standalone_question": text.strip().strip('"'), "attempts": 0}


def retrieve(state: State) -> dict:
    """Embed the standalone question and take the four closest passages."""
    tool = _policy_tool()
    passages = tool.search_passages(state["standalone_question"], k=4)
    return {"retrieved": passages}


def _records(passages: list[Passage]) -> str:
    blocks = []
    for index, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{index}] doc_id={passage['doc_id']} topic={passage['topic']}\n"
            f"reference={passage['reference']}\n"
            f"{passage['text']}"
        )
    return "\n\n".join(blocks)


def _parse_model(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)


def _grams(text: str, size: int = 4) -> set[str]:
    words = _flat(text).split()
    return {
        " ".join(words[i : i + size])
        for i in range(max(len(words) - size + 1, 0))
    }


def _answer_uses(answer: str, passage: Passage) -> bool:
    """True when the answer repeats a distinctive phrase from this passage."""
    answer_grams = _grams(answer)
    for gram in _grams(passage["text"]):
        if gram in answer_grams and any(len(word) > 6 for word in gram.split()):
            return True
    return False


def _with_source(citation: dict, passages: list[Passage]) -> Citation | None:
    quote = _flat(citation.get("quote", ""))
    if not quote:
        return None
    for passage in passages:
        if passage["doc_id"] == citation.get("doc_id") and quote in _flat(passage["text"]):
            return {
                "doc_id": passage["doc_id"],
                "quote": passage["text"],
                "topic": passage["topic"],
                "reference": passage["reference"],
            }
    return None


def _cover(answer: str, citations: list[Citation], passages: list[Passage]) -> list[Citation]:
    """Show the whole passage whenever the answer uses any part of it."""
    shown = {citation.get("quote", "") for citation in citations}
    covered = list(citations)
    for passage in passages:
        if passage["text"] in shown or not _answer_uses(answer, passage):
            continue
        covered.append(
            {
                "doc_id": passage["doc_id"],
                "quote": passage["text"],
                "topic": passage["topic"],
                "reference": passage["reference"],
            }
        )
        shown.add(passage["text"])
    return covered


def generate(state: State) -> dict:
    """Answer in the model's own words and cite a sentence copied from Chroma."""
    retrieved = state.get("retrieved") or []
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return {"refusal": True, "attempts": state.get("attempts", 0)}
    attempt = state.get("attempts", 0) + 1
    if not retrieved:
        return {"answer": REFUSAL, "citations": [], "refusal": True, "attempts": attempt}
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before asking the chat model.")

    note = state.get("verify_note") or ""
    correction = (
        f"\nYour previous quote failed verification: {note}\n"
        "Copy the full supporting span from the record, including any list items you used.\n"
        if note
        else ""
    )
    prompt = (
        "You answer questions about Northline Assurance claims manuals. "
        "Use only the records below. Do not invent a rule, a deadline, or an exception.\n"
        "Write the answer in your own words, but keep the manual's own timing words, "
        "dollar amounts, and defined terms. If a record says '1 business day' or "
        "'same business day', those words belong in the answer. A table row is a record.\n"
        "Each citation quote must be copied exactly from a single record, and it must "
        "include the sentence that states the rule you used. "
        "Never join two records into one quote. A sentence that ends in a colon and the "
        "list under it are different records: cite the sentence as one quote and the list "
        "items you use as a second quote. Do not refuse when a record states the rule.\n"
        "If no record states the rule the question asks for, set refusal to true and say "
        "the files do not contain that. A neighboring topic is not enough. Still cite "
        "the closest record so the reader can see it.\n"
        f"{correction}\n"
        f"Question: {state['standalone_question']}\n\n"
        f"Records:\n{_records(retrieved)}\n\n"
        'Return only JSON: {"answer": "...", "citations": [{"doc_id": "CLM-001", "quote": "exact words"}], "refusal": false}'
    )
    response = ChatOpenAI(model=CHAT_MODEL, temperature=0).invoke(prompt)
    content = response.content if isinstance(response.content, str) else str(response.content)
    try:
        parsed = _parse_model(content)
    except json.JSONDecodeError:
        return {
            "answer": "",
            "citations": [{"doc_id": "", "quote": ""}],
            "refusal": False,
            "attempts": attempt,
        }
    citations = [
        item
        for item in (
            _with_source(raw, retrieved)
            for raw in parsed.get("citations") or []
            if isinstance(raw, dict)
        )
        if item is not None
    ]
    answer = str(parsed.get("answer") or "").strip()
    return {
        "answer": answer,
        "citations": _cover(answer, citations, retrieved),
        "refusal": bool(parsed.get("refusal")),
        "attempts": attempt,
        "verify_note": "",
    }


def verify(state: State) -> dict:
    """Accept a citation only when its quote is inside a retrieved passage."""
    citations = state.get("citations") or []
    if state.get("refusal") and not citations:
        return {"citation_ok": True, "verify_note": ""}
    passages: dict[str, list[str]] = {}
    for passage in state.get("retrieved") or []:
        passages.setdefault(passage["doc_id"], []).append(_flat(passage["text"]))
    for citation in citations:
        quote = _flat(citation.get("quote", ""))
        texts = passages.get(citation.get("doc_id", ""), [])
        if not quote or not any(quote in text for text in texts):
            return {
                "citation_ok": False,
                "verify_note": "That quote is not an exact span of the cited record.",
            }
    return {"citation_ok": bool(citations), "verify_note": ""}


def refuse(state: State) -> dict:
    """Keep the closest passage visible after the quote fails five times."""
    top = (state.get("retrieved") or [None])[0]
    citations = []
    if top:
        citations = [{
            "doc_id": top["doc_id"],
            "quote": top["text"],
            "topic": top["topic"],
            "reference": top["reference"],
        }]
    return {"answer": REFUSAL, "citations": citations, "refusal": True, "citation_ok": True}


def format_result(result: State) -> str:
    """Model words, then every quote, then one citation for each source."""
    lines = [result.get("answer", "").strip(), ""]
    label = "Closest passage" if result.get("refusal") else "Quote"
    groups: dict[tuple[str, str, str], list[str]] = {}
    order: list[tuple[str, str, str]] = []
    for citation in result.get("citations") or []:
        key = (
            citation.get("doc_id", ""),
            citation.get("topic", ""),
            citation.get("reference", ""),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        quote = citation.get("quote", "").strip()
        if quote and quote not in groups[key]:
            groups[key].append(quote)
    for doc_id, topic, reference in order:
        lines.append(label)
        lines.append("")
        lines.extend(groups[(doc_id, topic, reference)])
        lines.append("")
        lines.append("Citation")
        heading = f"{doc_id} — {topic}" if topic else doc_id
        lines.append(heading)
        if reference:
            lines.append(reference)
        lines.append("")
    return "\n".join(lines).strip()


def route_after_verify(state: State) -> Literal["generate", "refuse", "__end__"]:
    if state.get("citation_ok"):
        return END
    if state.get("attempts", 0) < MAX_ATTEMPTS:
        return "generate"
    return "refuse"


def run_config(thread_id: str) -> dict:
    """Thread id plus a step ceiling so a routing bug cannot call the model forever."""
    return {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }


def build_graph():
    """Compile the sequential graph. The checkpointer keeps one thread in memory."""
    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("refuse", END)
    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]).strip() or "How fast must an appointed agent forward a notice of loss?"
    app = build_graph()
    result = app.invoke(
        {"question": question, "history": [], "attempts": 0},
        config=run_config("starter"),
    )
    print(format_result(result))
