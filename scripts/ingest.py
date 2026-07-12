"""Ingest knowledge/*.md into a persisted chroma collection. One card = one
chunk (cards are short, ~20 lines) -- per TOOLKIT_SPEC.md, repo 22's recipe.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cache import embed

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


def _parse_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else path.stem
    return {"id": path.stem, "title": title, "text": text}


def build() -> int:
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection("knowledge")
    except Exception:
        pass
    coll = client.create_collection("knowledge")

    cards = [_parse_card(p) for p in sorted(KNOWLEDGE_DIR.glob("*.md"))]
    ids = [c["id"] for c in cards]
    docs = [c["text"] for c in cards]
    metas = [{"title": c["title"]} for c in cards]
    embeddings = [embed(d).tolist() for d in docs]

    coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(cards)


if __name__ == "__main__":
    n = build()
    print(f"Ingested {n} knowledge cards into {CHROMA_DIR}")
