# app/packer.py
from typing import List, Optional, Dict
from .models import Evidence
from .config import CTX_BUDGET_TOKENS, EVIDENCE_HEADROOM

def _rough_tokens(s: str) -> int:
    # грубая оценка: ~4 символа на токен
    return max(1, len(s) // 4)

def _evidence_block_len(e: Evidence) -> int:
    # оценка токенов одного блока
    head = f"[{e.doc['id']} п.{e.loc.get('chunk_rule_number') or '?'}] {e.doc['title']} (ред. {e.doc['version']} от {e.doc['effective_date']})\n"
    return _rough_tokens(head) + _rough_tokens(e.text) + 8

def pack_evidence(anchor: Evidence,
                  neighbors: List[Evidence],
                  glossary: List[Evidence],
                  refs: List[Evidence]) -> List[Evidence]:
    """
    Приоритет включения: anchor → neighbors → glossary → refs.
    Влезаем в бюджет: CTX_BUDGET_TOKENS - EVIDENCE_HEADROOM (но не меньше 512).
    """
    budget = max(512, CTX_BUDGET_TOKENS - EVIDENCE_HEADROOM)
    out: List[Evidence] = []
    used = 0

    def _try_add(lst: List[Evidence], limit: int | None = None):
        nonlocal used, out
        for i, e in enumerate(lst):
            if limit is not None and i >= limit:
                break
            need = _evidence_block_len(e)
            if used + need > budget:
                break
            out.append(e)
            used += need

    _try_add([anchor], 1)
    _try_add(neighbors, 5)   # чуть короче окна = меньше оффтопа
    _try_add(glossary, 4)
    _try_add(refs, 3)
    return out

def render_system_prompt() -> str:
    return (
        "Ты юридический ассистент КТЖ. Отвечай ТОЛЬКО на основании предоставленных фрагментов (evidence). "
        "Каждое предложение ДОЛЖНО заканчиваться ссылкой вида [<doc_id> п.<chunk_rule_number>]. "
        "Если в evidence нет прямой нормы — явно напиши: «По прямой норме нет данных в предоставленных фрагментах» и попроси указать пункт/документ. "
        "Запрещены домыслы и выводы вне текста evidence. Используй короткие, юридически точные формулировки; ключевые фразы можно брать в кавычки."
        "Если ответ опирается на несколько подпунктов, объединяй их в один абзац, но ставь ОТДЕЛЬНУЮ ссылку после каждого утверждения/цифры."
    )

def render_user_prompt(question: str, evidences: List[Evidence], history: Optional[List[Dict]] = None) -> str:
    blocks = []
    for e in evidences:
        ref = f"[{e.doc['id']} п.{e.loc.get('chunk_rule_number') or '?'}]"
        header = f"{ref} {e.doc['title']} (ред. {e.doc['version']} от {e.doc['effective_date']})"
        blocks.append(f"{header}\n{e.text}")
    evid_text = "\n\n---\n\n".join(blocks)

    hist_text = ""
    if history:
        last = history[-10:]
        lines = []
        for turn in last:
            role = turn.get("role","user")
            content = (turn.get("content") or "")[:500]
            lines.append(f"{role.upper()}: {content}")
        hist_text = "Контекст предыдущего диалога:\n" + "\n".join(lines) + "\n\n"

    return (
        f"{hist_text}"
        f"Вопрос: {question}\n\nНормативные фрагменты:\n\n{evid_text}\n\n"
        "Сформируй ответ строго по тексту фрагментов. "
        "Каждое ПРЕДЛОЖЕНИЕ обязано заканчиваться ссылкой [<doc_id> п.<chunk_rule_number>]. "
        "Если нет прямой нормы — напиши об этом явно и не отвечай по общим соображениям."
    )
