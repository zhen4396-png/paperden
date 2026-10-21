# paper-rag-lite

> 一个**极简、可本地跑通**的论文 RAG（检索增强生成）小工具。
> 上传 PDF / TXT 论文 → 本地向量库检索 → 本地大模型回答。
> **论文不出本机，不上传任何云端。**

灵感来自 GitHub 上热门的 RAGFlow / Ollama / ChromaDB 方向，但砍掉了 90% 的复杂度，
只保留核心链路，方便学习 RAG 原理和自用。

---

## ✨ 特性

- 📚 **支持 PDF / TXT**：拖拽上传，自动分块入库
- 🔒 **全本地**：向量库用 ChromaDB（嵌入式），LLM 走本地 Ollama，零云端调用
- 🧠 **离线 embedding**：首次运行自动下载 all-MiniLM（~80MB），之后完全离线
- 🪶 **极简代码**：核心逻辑不到 200 行 Python，适合阅读和二次开发
- 🛟 **优雅降级**：没装 Ollama 也能用——自动切换成"只检索、展示原文片段"模式

---

## 📁 项目结构

```
paper-rag-lite/
├── app.py              # Streamlit 主界面
├── requirements.txt
├── rag/
│   ├── ingest.py       # PDF/TXT 解析、分块、入库
│   ├── retrieve.py     # 向量检索
│   └── llm.py          # Ollama 调用 + 降级模式
├── data/               # ChromaDB 持久化目录（自动生成，不入库）
└── demo/
    └── sample_paper.txt  # 样例文本，用于快速体验
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动

```bash
streamlit run app.py
```

浏览器自动打开 http://localhost:8501

### 3. 体验

1. 左侧上传 `demo/sample_paper.txt`（或你自己的 PDF）
2. 点"导入到知识库"
3. 右侧输入问题，回车查看回答和引用片段

---

## 🧠 接上本地大模型（可选）

默认模式下不调用 LLM，只展示检索到的原文片段。
想让它真正"生成答案"，装一个 [Ollama](https://ollama.com/)：

```bash
# 安装 Ollama 后，拉一个 3B 小模型（约 2GB）
ollama pull qwen2.5:3b
```

Ollama 默认跑在 `http://localhost:11434`，本工具会自动检测。
侧边栏出现绿色"Ollama 已连接"后，提问就会由本地模型生成答案。

> 中文效果更好可以换 `qwen2.5:7b` 或 `qwen3:4b`，在侧边栏下拉选择。

---

## 🔧 自定义

| 想改什么 | 改哪里 |
|---|---|
| 分块大小 / 重叠 | `rag/ingest.py` 的 `_chunk_text()` |
| 检索返回条数 | `app.py` 里 `retrieve.retrieve(q, top_k=4)` |
| 默认模型 | `rag/llm.py` 的 `DEFAULT_MODEL` |
| 中文检索更准 | 把 embedding 换成 `BAAI/bge-small-zh-v1.5`（需 `pip install sentence-transformers`） |

---

## 📝 原理（30 秒版）

```
PDF/TXT → 切分成 ~800 字片段 → 每个片段算向量 → 存进 ChromaDB
                                                          ↓
你的问题 → 算向量 → 在 ChromaDB 里找最相似的 4 个片段 → 拼进 prompt → Ollama 生成答案
```

这就是 RAG 的最小闭环。

---

## 📄 License

MIT
