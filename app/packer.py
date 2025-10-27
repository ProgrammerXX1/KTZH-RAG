from typing import List
from .models import Evidence   # ← должно стоять перед использованием в сигнатурах функций
from .config import CTX_BUDGET_TOKENS, EVIDENCE_HEADROOM
def _rough_tokens(s: str) -> int:
    # грубая оценка: ~4 символа на токен
    return max(1, len(s) // 4)

def _evidence_block_len(e: Evidence) -> int:
    # прикидка токенов одного блока в prompt'е
    head = f"[{e.doc['id']} п.{e.loc.get('chunk_rule_number') or '?'}] {e.doc['title']} (ред. {e.doc['version']} от {e.doc['effective_date']})\n"
    return _rough_tokens(head) + _rough_tokens(e.text) + 8  # небольшой оверхед

def pack_evidence(anchor: Evidence,
                  neighbors: List[Evidence],
                  glossary: List[Evidence],
                  refs: List[Evidence]) -> List[Evidence]:
    # Приоритет: anchor → neighbors → glossary → refs, но вписываемся в токен-бюджет
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
            out.append(e); used += need

    _try_add([anchor], 1)
    _try_add(neighbors, 4)
    _try_add(glossary, 3)
    _try_add(refs, 2)
    return out
def render_system_prompt() -> str:
    return (
        "Ты юридический ассистент КТЖ. Отвечай строго на основании приведённых фрагментов (evidence). "
        "Всегда ставь ссылки в формате [<doc_id> п.<chunk_rule_number>] по каждому утверждению. "
        "Если прямой нормы нет — так и скажи, какие документы/разделы нужны. "
        "Не придумывай числа и единицы."
    )

def render_user_prompt(question: str, evidences: List[Evidence]) -> str:
    blocks = []
    for e in evidences:
        ref = f"[{e.doc['id']} п.{e.loc.get('chunk_rule_number') or '?'}]"
        header = f"{ref} {e.doc['title']} (ред. {e.doc['version']} от {e.doc['effective_date']})"
        blocks.append(f"{header}\n{e.text}")
    evid_text = "\n\n---\n\n".join(blocks)
    return f"Вопрос: {question}\n\nНормативные фрагменты:\n\n{evid_text}\n\nСформируй ответ со ссылками."
