"""Query the ingested knowledge chroma collection."""
from __future__ import annotations

from pathlib import Path

from cache import embed

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection("knowledge")
    return _collection


def retrieve(query: str, k: int = 6) -> list[dict]:
    """Returns [{id, title, text}] ranked by relevance to `query`."""
    coll = _get_collection()
    result = coll.query(query_embeddings=[embed(query).tolist()], n_results=k)
    cards = []
    for i in range(len(result["ids"][0])):
        cards.append({
            "id": result["ids"][0][i],
            "title": result["metadatas"][0][i]["title"],
            "text": result["documents"][0][i],
        })
    return cards
