from typing import List
from .models import Evidence
from .config import CTX_BUDGET_TOKENS, EVIDENCE_HEADROOM

# --- токен-бюджет ---
def _rough_tokens(s: str) -> int:
    # грубая оценка: ~4 символа на токен
    return max(1, len(s) // 4)

def _fmt_header(e: Evidence) -> str:
    doc = e.doc or {}
    loc = e.loc or {}
    ref = f"[{doc.get('id') or '?'} п.{loc.get('chunk_rule_number') or '?'}]"
    title = doc.get("title") or "Без названия"
    version = doc.get("version") or "?"
    eff = doc.get("effective_date") or "?"
    return f"{ref} {title} (ред. {version} от {eff})\n"

def _truncate_sentences(text: str, max_chars: int = 600) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # обрежем по последнему завершителю, если он не слишком близко к началу
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last >= max(120, int(max_chars * 0.4)):
        return cut[: last + 1]
    return cut

def _evidence_block_len(e: Evidence) -> int:
    head = _fmt_header(e)
    return _rough_tokens(head) + _rough_tokens(e.text) + 8

def _cap_by_budget(budget_tokens: int) -> tuple[int, int, int, int]:
    """
    Возвращает лимиты (anchor, neighbors, glossary, refs) по доступному бюджету.
    Агрессивно режем соседей при маленьком бюджете.
    """
    if budget_tokens < 1200:
        return (1, 2, 2, 1)
    if budget_tokens < 2400:
        return (1, 3, 3, 2)
    return (1, 4, 3, 2)

def pack_evidence(anchor: Evidence,
                  neighbors: List[Evidence],
                  glossary: List[Evidence],
                  refs: List[Evidence]) -> List[Evidence]:
    # бюджет на контекст
    budget = max(512, CTX_BUDGET_TOKENS - EVIDENCE_HEADROOM)
    a_cap, n_cap, g_cap, r_cap = _cap_by_budget(budget)

    # нормализуем тексты заранее
    def _prep(lst: List[Evidence]) -> List[Evidence]:
        out = []
        for e in lst:
            e = Evidence(**e.model_dump())  # копия
            e.text = _truncate_sentences(e.text or "", 600)
            out.append(e)
        return out

    anchor_p = _prep([anchor])
    neighbors_p = _prep(neighbors)[:n_cap]
    glossary_p = _prep(glossary)[:g_cap]
    refs_p = _prep(refs)[:r_cap]

    # сбор с дедупом по cite
    used = 0
    out: List[Evidence] = []
    seen = set()

    def _try_add(lst: List[Evidence], limit: int):
        nonlocal used
        added = 0
        for e in lst:
            if limit and added >= limit:
                break
            if e.cite in seen:
                continue
            need = _evidence_block_len(e)
            if used + need > budget:
                break
            out.append(e)
            seen.add(e.cite)
            used += need
            added += 1

    _try_add(anchor_p, a_cap)
    _try_add(neighbors_p, n_cap)
    _try_add(glossary_p, g_cap)
    _try_add(refs_p, r_cap)
    return out

def render_system_prompt() -> str:
    return (
        "Ты юридический ассистент КТЖ. Отвечай только на основании приведённых фрагментов (evidence). "
        "Ставь ссылки в формате [<doc_id> п.<chunk_rule_number>] к каждому утверждению. "
        "Если прямой нормы нет, укажи, каких документов/разделов не хватает, и не придумывай значения."
    )

def render_user_prompt(question: str, evidences: List[Evidence]) -> str:
    blocks = []
    for e in evidences:
        header = _fmt_header(e)
        blocks.append(f"{header}{e.text}")
    evid_text = "\n\n---\n\n".join(blocks)
    return (
        f"Вопрос: {question}\n\n"
        f"Нормативные фрагменты:\n\n{evid_text}\n\n"
        f"Сформируй ответ со ссылками по каждому факту."
    )
