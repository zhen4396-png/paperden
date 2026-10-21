"""一次性查询：直接调用后端检索，把结果打印出来。"""
from rag import retrieve

q = "书稿格式有哪些要求？"
print(f"问题：{q}\n" + "=" * 60)
ctx = retrieve.retrieve(q, top_k=4)
for i, c in enumerate(ctx, 1):
    print(f"\n【片段 {i}】来源：{c['source']} · 第 {c['page']} 页 · 相关度 {c['score']}")
    print("-" * 60)
    print(c["text"])
