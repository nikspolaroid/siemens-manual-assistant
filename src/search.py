"""Semantic search over the manual index. No LLM involved."""
import sys

from config import TOP_K
from db import get_collection, embed


def search(question: str, k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("Index is empty. Run: python src/index.py")

    qv = embed([question])
    res = collection.query(query_embeddings=qv, n_results=k)

    return [
        {"text": doc, "source": meta["source"],
         "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What causes DC link overvoltage?"
    print(f"Q: {q}\n")
    for i, h in enumerate(search(q), 1):
        preview = h["text"][:220].replace("\n", " ")
        print(f"[{i}] {h['source']} p.{h['page']}  (distance {h['distance']:.3f})")
        print(f"    {preview}...\n")
