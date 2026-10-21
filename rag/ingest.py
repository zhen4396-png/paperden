"""PDF / TXT 解析、分块、入库到 ChromaDB。"""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import chromadb
from pypdf import PdfReader

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "papers"


def get_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    client = get_client()
    # 默认 embedding：Chroma 自带的 all-MiniLM（ONNX），首次运行自动下载约 80MB
    # 用余弦距离，相似度 = 1 - dist，落在 [-1, 1]，越大越相关
    return client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """极简分块：按字符长度切，带 overlap。中文按字符切也够用。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    reader = PdfReader(BytesIO(file_bytes))
    col = get_collection()
    added = 0
    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for chunk in _chunk_text(text):
            if len(chunk) < 50:  # 跳过过短碎片
                continue
            col.add(
                ids=[f"{filename}_p{page_idx}_chunk{added}"],
                documents=[chunk],
                metadatas=[{"source": filename, "page": page_idx + 1}],
            )
            added += 1
    return added


def ingest_text(text: str, filename: str = "inline.txt") -> int:
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
    return get_collection().count()


def reset_db() -> None:
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
