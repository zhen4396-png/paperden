"""冒烟测试：不启动 Streamlit，直接验证 ingest + retrieve 链路。"""
from pathlib import Path

from rag import ingest, llm, retrieve

ROOT = Path(__file__).resolve().parent
sample = (ROOT / "demo" / "sample_paper.txt").read_text(encoding="utf-8")

# 清空旧库，保证测试干净
ingest.reset_db()
n = ingest.ingest_text(sample, filename="sample_paper.txt")
print(f"[OK] 入库片段数：{n}，库内总数：{ingest.count_chunks()}")

print(f"[INFO] Ollama 是否在线：{llm.is_ollama_alive()}")

for q in [
    "最优沟槽间距是多少？",
    "计算用的湍流模型是什么？",
    "沟槽间距太大时会发生什么？",
]:
    ctx = retrieve.retrieve(q, top_k=2)
    print(f"\n=== Q: {q} ===")
    for c in ctx:
        print(f"  - [{c['source']} p{c['page']}] score={c['score']}")
        print(f"    {c['text'][:120]}...")
