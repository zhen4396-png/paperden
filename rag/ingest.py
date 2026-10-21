"""PDF / TXT 解析、分块、入库到 ChromaDB。

这是 RAG 流程的"写入端"：
  PDF/TXT → 提取纯文本 → 切成 ~800 字片段 → 算向量 → 存进本地向量库
"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import chromadb
from pypdf import PdfReader

# 向量库持久化目录：项目根目录/data/chroma/，已在 .gitignore 中排除
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "papers"  # ChromaDB 里的"表名"


def get_client() -> chromadb.ClientAPI:
    """创建/获取一个持久化的 ChromaDB 客户端。"""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    """获取向量集合（不存在就新建）。

    - embedding 用 Chroma 自带的 all-MiniLM（ONNX），首次运行自动下载约 80MB
    - 距离度量用余弦距离：相似度 = 1 - dist，范围 [-1, 1]，越大越相关
    """
    client = get_client()
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """把长文本切成带重叠的片段。

    为什么需要 overlap？避免把一句话从中间切断，相邻片段留 150 字重叠，
    保证检索时上下文不丢。
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap  # 步长 = 块长 - 重叠
    return chunks


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """解析 PDF 文件并入库，返回写入的片段数。"""
    reader = PdfReader(BytesIO(file_bytes))
    col = get_collection()
    added = 0
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in _chunk_text(text):
            if len(chunk) < 50:  # 跳过页眉页脚等过短碎片
                continue
            col.add(
                ids=[f"{filename}_p{page_idx}_chunk{added}"],
                documents=[chunk],
                metadatas=[{"source": filename, "page": page_idx + 1}],
            )
            added += 1
    return added


def ingest_text(text: str, filename: str = "inline.txt") -> int:
    """直接把一段文本入库（用于 TXT 文件或快速测试）。"""
    col = get_collection()
    added = 0
    for chunk in _chunk_text(text):
        if len(chunk) < 50:
            continue
        col.add(
            ids=[f"{filename}_chunk{added}"],
            documents=[chunk],
            metadatas=[{"source": filename, "page": 0}],
        )
        added += 1
    return added


def count_chunks() -> int:
    """当前库里总共有多少个片段。"""
    return get_collection().count()


def reset_db() -> None:
    """清空整个向量库（删除集合后自动重建空集合）。"""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
