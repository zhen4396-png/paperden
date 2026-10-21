"""本地 LLM 调用：优先走 Ollama；没装就降级为"只检索、不生成"模式。"""
from __future__ import annotations

import requests

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


def is_ollama_alive() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_local_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def answer(query: str, contexts: list[dict], model: str = DEFAULT_MODEL) -> tuple[str, bool]:
    """返回 (回答文本, 是否真正调用了本地LLM)。"""
    ctx = "\n\n---\n\n".join(
        [f"[{c['source']} p{c['page']}]\n{c['text']}" for c in contexts]
    )
    prompt = (
        "你是一个严谨的学术论文助手。请只根据下面给出的论文片段回答用户问题。"
        "不要编造片段中没有的数据或结论；如果片段里没有答案，直接说\"资料中未提及\"。\n\n"
        f"论文片段：\n{ctx}\n\n"
        f"用户问题：{query}\n\n回答："
    )
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip(), True
    except Exception:
        # 降级：不调 LLM，直接把检索到的原文片段展示给用户
        out = "⚠️ 未连接到本地 Ollama，下面直接展示检索到的原文片段：\n\n"
        for c in contexts:
            out += f"📄 【{c['source']} p{c['page']}】相关度 {c['score']}\n{c['text']}\n\n"
        return out, False
