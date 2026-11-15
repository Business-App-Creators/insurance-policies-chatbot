"""Score the 14-question key from the cited-chat plan.

    python3 -m chatbot.eval.score

A question passes when the expected manual is in the four retrieved passages,
the quote check passed, and every required fact appears in the reply.
Q12 and Q13 pass when the assistant says the files do not contain that.
Q14 also requires the rewritten question to mention the earlier topic.
"""

import json
import uuid
from pathlib import Path

from chatbot.agents.sequential.graph import build_graph, format_result

QUESTIONS = Path(__file__).resolve().parent / "questions.json"
PASS_BAR = 12


def _flat(text: str) -> str:
    return " ".join(text.split()).casefold()


def _visible(result: dict) -> str:
    parts = [result.get("answer") or ""]
    parts.extend(citation.get("quote") or "" for citation in result.get("citations") or [])
    return _flat("\n".join(parts))


def _ask(graph, item: dict) -> dict:
    thread = str(uuid.uuid4())
    history = []
    if item.get("history_question"):
        first = graph.invoke(
            {"question": item["history_question"], "history": [], "attempts": 0},
            config={"configurable": {"thread_id": thread + "-setup"}},
        )
        history = [
            {"role": "user", "content": item["history_question"]},
            {"role": "assistant", "content": first.get("answer") or ""},
        ]
    return graph.invoke(
        {"question": item["question"], "history": history, "attempts": 0},
        config={"configurable": {"thread_id": thread}},
    )


def _score(item: dict, result: dict) -> dict:
    retrieved = [passage.get("doc_id") for passage in result.get("retrieved") or []]
    expected = item.get("doc")
    retrieval = expected in retrieved if expected else None
    facts = item.get("fact") or []
    visible = _visible(result)
    fact_ok = all(fact.casefold() in visible for fact in facts)
    citation_ok = bool(result.get("citation_ok"))
    refusal = bool(result.get("refusal"))
    standalone = _flat(result.get("standalone_question") or "")
    mentions = item.get("standalone_must_mention") or []
    rewrite_ok = all(term.casefold() in standalone for term in mentions) if mentions else True
    if expected is None:
        passed = refusal and "do not contain" in _flat(result.get("answer") or "")
    else:
        passed = retrieval and citation_ok and fact_ok and not refusal and rewrite_ok
    return {
        "id": item["id"],
        "retrieval": retrieval,
        "citation": citation_ok,
        "fact": fact_ok,
        "refusal": refusal,
        "rewrite": rewrite_ok,
        "passed": passed,
        "top": retrieved,
        "standalone": result.get("standalone_question") or "",
    }


def main() -> None:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    graph = build_graph()
    rows = []
    for item in questions:
        print(f"running {item['id']}…", flush=True)
        rows.append(_score(item, _ask(graph, item)))

    retrieval_hits = sum(row["retrieval"] is True for row in rows)
    retrieval_expected = sum(row["retrieval"] is not None for row in rows)
    citation_passed = sum(row["citation"] for row in rows if row["retrieval"] is not None)
    facts_correct = sum(row["fact"] for row in rows if row["retrieval"] is not None)
    refusals_correct = sum(row["passed"] for row in rows if row["retrieval"] is None)
    passed = sum(row["passed"] for row in rows)

    print()
    print(f"{'id':<5} {'pass':<6} {'doc':<8} {'cite':<6} {'fact':<6} {'refuse':<7} top")
    for row in rows:
        doc = "yes" if row["retrieval"] else ("n/a" if row["retrieval"] is None else "no")
        print(
            f"{row['id']:<5} {str(row['passed']):<6} {doc:<8} "
            f"{str(row['citation']):<6} {str(row['fact']):<6} {str(row['refusal']):<7} "
            f"{', '.join(row['top'])}"
        )
        if row["id"] == "Q14":
            print(f"      standalone: {row['standalone']}")

    print()
    print(f"retrieval hits: {retrieval_hits} of {retrieval_expected}")
    print(f"citation checks passed: {citation_passed} of {retrieval_expected}")
    print(f"facts correct: {facts_correct} of {retrieval_expected}")
    print(f"refusals correct: {refusals_correct} of {len(rows) - retrieval_expected}")
    print(f"all checks: {passed} of {len(rows)} (bar is {PASS_BAR})")
    if passed < PASS_BAR:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
