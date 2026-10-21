"""向量检索：根据问题返回最相关的若干片段。

这是 RAG 流程的"读取端"：
  用户问题 → 算向量 → 在 ChromaDB 里找最相似的 top_k 个片段 → 返回给 LLM
"""
from __future__ import annotations

from .ingest import get_collection


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """根据问题检索最相关的片段。

    Args:
        query: 用户的自然语言问题
        top_k: 返回前几个最相关的片段

    Returns:
        每个片段包含：
          - text: 片段原文
          - source: 来源文件名
          - page: 页码
          - score: 相关度（0~1，越大越相关）
    """
    col = get_collection()
    if col.count() == 0:
        return []
    k = min(top_k, col.count())
    res = col.query(query_texts=[query], n_results=k)

    out: list[dict] = []
    # ChromaDB 返回的是嵌套列表（query 维度在最外层），取第一组结果
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        out.append(
            {
                "text": doc,
                "source": meta.get("source", "?"),
                "page": meta.get("page", 0),
                "score": round(1 - dist, 3),  # 余弦距离 → 相似度，越大越相关
            }
        )
    return out
