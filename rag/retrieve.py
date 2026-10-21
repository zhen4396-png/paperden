"""向量检索：根据问题返回最相关的若干片段。"""
from __future__ import annotations

from .ingest import get_collection


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    k = min(top_k, col.count())
    res = col.query(query_texts=[query], n_results=k)
    out: list[dict] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "page": meta.get("page", 0),
                "score": round(1 - dist, 3),  # 越大越相关
            }
        )
    return out
