# app/rewriter.py
from __future__ import annotations
import json, re, asyncio
from typing import Dict, List
from .llm import chat

# простая нормализация до LLM
_space_re = re.compile(r"\s+", re.U)
def _preclean(q: str) -> str:
    q = (q or "").replace("\x00", " ").strip()
    q = q.replace("пп.", "п.").replace("пунктов", "п.").replace("пункта", "п.").replace("пункты", "п.")
    q = q.replace("статья", "ст.").replace("статьи", "ст.")
    q = _space_re.sub(" ", q)
    # нормализация «п. 3.5.» -> «п. 3.5»
    q = re.sub(r"(п\.?\s*)(\d+(?:\.\d+)*)(\.)\b", r"\1\2", q, flags=re.I)
    return q[:1000]

SYSTEM = (
    "Ты помогаешь нормализовать юридические запросы к корпоративным документам КТЖ. "
    "Выведи JSON без пояснений:\n"
    "{"
    "\"canonical\": string,           # краткий канонический запрос для поиска\n"
    "\"expansions\": [string, ...]    # до 3 альтернатив синонимичных формулировок\n"
    "}\n"
    "Правила: сохраняй упомянутые номера пунктов/подпунктов, даты и идентификаторы документов (например, 854-ЦЗ, 291-ИДП). "
    "Убирай лишние слова, просторечие и эмоции. Пиши лаконично."
)

def _fallback_rule(q: str) -> Dict[str, List[str] | str]:
    # если LLM сломался — даём минимальный детерминизм
    q2 = _preclean(q.lower())
    # эвристические расширения
    ex: List[str] = []
    if any(t in q2 for t in ("что такое", "определение", "что означает", "расшифровка")):
        ex += ["определение термина", "что означает термин", "расшифровка сокращения"]
    ex = list(dict.fromkeys([e for e in ex if e not in q2]))[:3]
    return {"canonical": q2, "expansions": ex}

async def rewrite_query(raw: str, temperature: float = 0.1, max_tokens: int = 256) -> Dict[str, List[str] | str]:
    src = _preclean(raw)
    user = f"Запрос: {src}\nОтвети строго JSON как в инструкции."
    try:
        out = await chat(SYSTEM, user, temperature=temperature, max_tokens=max_tokens)
        j = json.loads(out)
        can = _preclean(j.get("canonical") or src)
        exs = [ _preclean(x) for x in (j.get("expansions") or []) if isinstance(x, str) ]
        exs = [e for e in exs if e and e != can][:3]
        return {"canonical": can, "expansions": exs}
    except Exception:
        return _fallback_rule(src)
