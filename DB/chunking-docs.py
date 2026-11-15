"""Split each policy manual into paragraph chunks.

Each chunk keeps the section it came from as its topic, and a reference back to
the original file. Definition paragraphs use the defined term as the topic.

    python DB/chunking-docs.py

Writes DB/chunks.json.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES_DIR = ROOT / "policies-documents"
CHUNKS_PATH = Path(__file__).resolve().parent / "chunks.json"

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DEFINITION = re.compile(r"^\*\*(.+?)\.\*\*")


def chunk_document(path: Path) -> list[dict]:
    """Return paragraph chunks for one markdown policy."""
    doc_id = "-".join(path.stem.split("-")[:2])
    reference = path.relative_to(ROOT).as_posix()
    title = ""
    section = ""
    chunks = []

    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        paragraph = block.strip()
        if not paragraph:
            continue

        lines = paragraph.splitlines()
        heading = HEADING.match(lines[0]) if len(lines) == 1 else None
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1:
                title = text
            section = text
            continue

        if paragraph.startswith("| Field | Value |"):
            topic = "Document control"
        else:
            defined = DEFINITION.match(paragraph)
            topic = defined.group(1).strip() if defined else (section or title or doc_id)

        chunks.append(
            {
                "id": f"{doc_id}-{len(chunks) + 1:04d}",
                "topic": topic,
                "section": section or title,
                "text": paragraph,
                "reference": reference,
                "doc_id": doc_id,
                "title": title,
            }
        )

    return chunks


def chunk_policies(policies_dir: Path = POLICIES_DIR) -> list[dict]:
    """Chunk every policy manual, in filename order."""
    paths = sorted(policies_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No policy documents in {policies_dir}")
    chunks = []
    for path in paths:
        chunks.extend(chunk_document(path))
    return chunks


def main() -> None:
    chunks = chunk_policies()
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    documents = sorted({chunk["doc_id"] for chunk in chunks})
    print(f"Wrote {len(chunks)} chunks from {len(documents)} documents to {CHUNKS_PATH}")
    for doc_id in documents:
        count = sum(chunk["doc_id"] == doc_id for chunk in chunks)
        print(f"  {doc_id}: {count}")


if __name__ == "__main__":
    main()
