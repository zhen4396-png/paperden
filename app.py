"""paper-rag-lite：本地论文 RAG 小工具（Streamlit 界面）。"""
from __future__ import annotations

import streamlit as st

from rag import ingest, llm, retrieve

st.set_page_config(page_title="paper-rag-lite", page_icon="📚", layout="wide")
st.title("📚 paper-rag-lite")
st.caption("本地论文检索问答 · 向量库 + Ollama · 论文不出本机")

# ---------- 侧边栏：入库 ----------
with st.sidebar:
    st.header("① 导入论文")
    uploaded = st.file_uploader("上传 PDF 或 TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded and st.button("导入到知识库", type="primary"):
        for f in uploaded:
            data = f.read()
            if f.name.lower().endswith(".pdf"):
                n = ingest.ingest_pdf(data, f.name)
            else:
                n = ingest.ingest_text(data.decode("utf-8", errors="ignore"), f.name)
            st.success(f"{f.name}：写入 {n} 个片段")
    st.divider()
    if st.button("🗑️ 清空知识库", use_container_width=True):
        ingest.reset_db()
        st.warning("已清空")

    st.divider()
    st.header("② 模型状态")
    if llm.is_ollama_alive():
        models = llm.list_local_models()
        st.success(f"Ollama 已连接，本地模型：{', '.join(models) if models else '（暂无，先 ollama pull）'}")
        model_name = st.selectbox("使用模型", models or [llm.DEFAULT_MODEL], index=0)
    else:
        st.error("未检测到 Ollama（http://localhost:11434）")
        st.caption("装了 Ollama 并 `ollama pull qwen2.5:3b` 后即可生成答案；否则只能看检索到的原文片段。")
        model_name = llm.DEFAULT_MODEL

# ---------- 主区：提问 ----------
st.info(f"知识库当前片段数：**{ingest.count_chunks()}**")

q = st.text_input("提问：", placeholder="例如：这篇论文用了什么湍流模型？雷诺数范围是多少？")
if q:
    ctx = retrieve.retrieve(q, top_k=4)
    if not ctx:
        st.warning("知识库里还没有内容，请先在左侧上传论文。")
    else:
        answer_text, used_llm = llm.answer(q, ctx, model=model_name)
        st.subheader("回答")
        if used_llm:
            st.write(answer_text)
        else:
            st.markdown(answer_text)
        with st.expander("📎 引用片段（可展开核对原文）"):
            for c in ctx:
                st.markdown(f"**{c['source']} · 第 {c['page']} 页**（相关度 {c['score']}）")
                st.write(c["text"])
                st.divider()
