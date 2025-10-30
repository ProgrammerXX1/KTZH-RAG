from __future__ import annotations

import datetime
import io
import json
import zipfile
from typing import Any, Dict, Iterator, List, Optional
import re
from collections import defaultdict, deque

from fastapi import FastAPI, File, UploadFile

from .llm import chat
from .models import AnswerRequest, Chunk
from .packer import pack_evidence, render_system_prompt, render_user_prompt
from .retriever import (
    build_section_window,
    chunk_types_overview,
    collections_counts,
    delete_by_source_document_id,
    ensure_schema,
    ingest_chunks,
    reset_schema,
    search,
    to_evidence,
    wipe_all,
    _search_collection,   # для подсказок
    RULES,                # для подсказок
    _doc_id_of,           # для подсказок
)

# допускаем точку/пробел после скобок, иначе добавлялся фолбэк
_CITE_RE = re.compile(r"\[[^\[\]]+ п\.[^\[\]]+\]\s*[\.!\?]?$")
HISTORY: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))

def _ensure_citations(text: str, fallback_cite: str) -> str:
    # если предложение уже заканчивается [doc п.X] (с опц. точкой) — ничего не добавляем
    lines = [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]
    out = []
    for s in lines:
        out.append(s if _CITE_RE.search(s) else (s + " " + fallback_cite))
    return " ".join(out)

# --- ZIP helpers (кириллица) ---
UTF8_FLAG = 0x800  # EFS flag

def _guess_cyr_name(name: str) -> str:
    try:
        raw = name.encode("cp437", errors="strict")
    except Exception:
        return name
    for enc in ("cp866", "cp1251"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return name

def _fix_zip_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & UTF8_FLAG:
        return info.filename
    return _guess_cyr_name(info.filename)

def _iter_jsonl_chunks(f: io.TextIOBase) -> Iterator[Chunk]:
    for _i, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            continue
        try:
            yield Chunk.model_validate_json(s)
        except Exception:
            continue

async def _ingest_streaming(file_like: io.TextIOBase, batch_size: int = 1000) -> int:
    buf: List[Chunk] = []
    total = 0
    for ch in _iter_jsonl_chunks(file_like):
        buf.append(ch)
        if len(buf) >= batch_size:
            total += await ingest_chunks(buf)
            buf.clear()
    if buf:
        total += await ingest_chunks(buf)
    return total

# --- Подсказки при отсутствии прямой нормы ---
async def _suggest_followups(query: str, language: Optional[str], today_iso: str, doc_id: Optional[str]) -> List[str]:
    lang = language or "ru"
    today_rfc3339 = f"{today_iso}T23:59:59Z"
    tips: List[str] = []
    try:
        from .retriever import _client
        with _client() as client:
            rows = await _search_collection(
                client.collections.get(RULES),
                query=query, k=30, language=lang, today_rfc3339=today_rfc3339,
                qv=None, alpha=0.0, doc_id=doc_id
            )
    except Exception:
        rows = []

    if not rows:
        return [
            "Уточните документ: 854-ЦЗ, 544-ПТЭ или 291-ИДП.",
            "Уточните пункт: например, «п. 3», «п. 3.1» или раздел/главу.",
        ]

    by_doc: Dict[str, List[dict]] = {}
    for r in rows:
        d = _doc_id_of(r) or "?"
        by_doc.setdefault(d, []).append(r)

    docs_sorted = sorted(by_doc.items(), key=lambda kv: len(kv[1]), reverse=True)[:2]
    for d, arr in docs_sorted:
        nums = []
        seen = set()
        for it in arr[:12]:
            n = (it.get("chunk_rule_number") or "").strip(".")
            if not n or n in seen:
                continue
            seen.add(n)
            nums.append(n)
        if nums:
            tips.append(f"Уточните пункт(ы) в {d}: " + ", ".join(nums[:3]))
        else:
            sec = (arr[0].get("section_number") or "?")
            tips.append(f"Уточните раздел в {d}: {sec}")

    tips.append("Уточните вид запроса: определение термина, требования или порядок действий.")
    return tips[:5]

# === FastAPI ===
app = FastAPI(title="KTZH RAG")

@app.on_event("startup")
def startup() -> None:
    ensure_schema()

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        return {"ok": True, "counts": collections_counts()}
    except Exception as e:
        return {"ok": True, "weaviate": "unavailable", "error": str(e)[:200]}

# ---------- Админ ----------
@app.get("/admin/stats")
def admin_stats() -> Dict[str, Any]:
    return {"counts": collections_counts()}

@app.delete("/admin/reset")
def admin_reset() -> Dict[str, Any]:
    reset_schema()
    return {"status": "ok", "action": "reset_schema", "counts": collections_counts()}

@app.delete("/admin/chunks/by-doc/{doc_id}")
def admin_delete_doc(doc_id: str) -> Dict[str, Any]:
    stats = delete_by_source_document_id(doc_id)
    return {"status": "ok", "deleted": stats, "counts": collections_counts()}

@app.delete("/admin/chunks/all")
def admin_delete_all() -> Dict[str, Any]:
    wipe_all()
    return {"status": "ok", "action": "wipe_all", "counts": collections_counts()}

@app.get("/admin/chunks/types")
def admin_chunk_types(limit_per_type: int = 3, doc_id: Optional[str] = None) -> Dict[str, Any]:
    overview = chunk_types_overview(limit_per_type=limit_per_type, doc_id=doc_id)
    return {"kinds": overview}

# ---------- Ингест ----------
@app.post("/ingest/zip")
async def ingest_zip(file: UploadFile = File(...)) -> Dict[str, Any]:
    data = await file.read()
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return {"error": "bad_zip_file", "filename": file.filename}

    total_ingested = 0
    per_file: List[Dict[str, Any]] = []

    for info in z.infolist():
        safe_name = _fix_zip_name(info)
        if info.is_dir():
            continue
        if not safe_name.lower().endswith(".jsonl"):
            continue
        try:
            with z.open(info, "r") as fbin:
                text_stream = io.TextIOWrapper(fbin, encoding="utf-8", errors="ignore")
                n = await _ingest_streaming(text_stream, batch_size=1000)
                per_file.append({"file": safe_name, "ingested": n})
                total_ingested += n
        except Exception as e:
            per_file.append({"file": safe_name, "error": str(e), "ingested": 0})

    return {
        "zip": file.filename,
        "total_ingested": total_ingested,
        "files": per_file,
        "counts": collections_counts(),
    }

@app.post("/ingest/jsonl/batch")
async def ingest_jsonl_batch(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    total_ingested = 0
    for uf in files:
        try:
            uf.file.seek(0)
            text_stream = io.TextIOWrapper(uf.file, encoding="utf-8", errors="ignore")
            n = await _ingest_streaming(text_stream, batch_size=1000)
            results.append({"file": uf.filename, "ingested": n})
            total_ingested += n
        except Exception as e:
            results.append({"file": uf.filename, "error": str(e), "ingested": 0})
    return {"total_ingested": total_ingested, "files": results, "counts": collections_counts()}

@app.post("/ingest/jsonl")
async def ingest_jsonl(file: UploadFile = File(...)) -> Dict[str, Any]:
    text = (await file.read()).decode("utf-8", "ignore")
    rows = [Chunk.model_validate_json(line) for line in text.splitlines() if line.strip()]
    n = await ingest_chunks(rows)
    return {"ingested": n, "counts": collections_counts()}

# ---------- RAG ----------
@app.post("/rag/answer")
async def rag_answer(req: AnswerRequest) -> Dict[str, Any]:
    today = datetime.date.today().isoformat()

    # История (до 10 реплик)
    hist = list(HISTORY[getattr(req, "conversation_id", "")]) if getattr(req, "conversation_id", None) else []

    # язык фиксированный, документ авто-выбором
    lang = "ru"
    forced_doc_id = None

    r = await search(
        req.query,
        req.k_rules,
        req.k_defs,
        language=lang,
        today=today,
        doc_id=forced_doc_id,
    )

    candidates = r.get("rules") or []
    if not candidates:
        defs = r.get("defs") or []
        abbr = r.get("abbr") or []
        pool = defs[:5] + abbr[:5]

        tips = await _suggest_followups(req.query, lang, today, forced_doc_id)

        if not pool and forced_doc_id:
            r2 = await search(
                req.query,
                req.k_rules,
                req.k_defs,
                language=lang,
                today=today,
                doc_id=forced_doc_id,
            )
            pool = (r2.get("defs") or []) + (r2.get("abbr") or [])

        if not pool:
            out = {
                "answer": "По прямой норме нет данных в предоставленных фрагментах. Используйте подсказки для уточнения запроса.",
                "evidence": [],
                "hints": tips,
            }
            if getattr(req, "conversation_id", None):
                HISTORY[req.conversation_id].append({"role": "user", "content": req.query})
                HISTORY[req.conversation_id].append({"role": "assistant", "content": out["answer"] + " " + " | ".join(tips)})
            return out

        from collections import Counter
        ids = [
            (json.loads(x.get("meta_json", "{}")).get("source_document_id") or x.get("source_document_id"))
            for x in pool
        ]
        if ids:
            top_id, _ = Counter([i for i in ids if i]).most_common(1)[0]
            pool = [
                x for x in pool
                if (json.loads(x.get("meta_json", "{}")).get("source_document_id") or x.get("source_document_id")) == top_id
            ]

        glossary_e = to_evidence(pool[:6], why="Глоссарий")
        user = render_user_prompt(req.query, glossary_e, history=hist)
        out_text = await chat(render_system_prompt(), user, temperature=0.0, max_tokens=800)
        fallback = f"[{glossary_e[0].doc['id']} п.{glossary_e[0].loc.get('chunk_rule_number') or '?'}]" if glossary_e else "[? п.?]"
        out_text = _ensure_citations(out_text, fallback)

        out = {"answer": out_text, "evidence": [e.model_dump() for e in glossary_e], "hints": tips}
        if getattr(req, "conversation_id", None):
            HISTORY[req.conversation_id].append({"role": "user", "content": req.query})
            HISTORY[req.conversation_id].append({"role": "assistant", "content": out_text})
        return out

    # якорь
    anchor_raw = max(candidates, key=lambda x: (x.get("_score") or 0.0))
    anchor_e = to_evidence([anchor_raw], why="Якорная норма")[0]

    # окно по разделу: убираем дубликаты «parent_rule», оставляем только подпункты
    window_raw = build_section_window(anchor_raw, neighbors=8)
    window_wo_anchor = [
        it for it in window_raw
        if it.get("chunk_id") != anchor_raw.get("chunk_id")
        and (it.get("chunk_type") or json.loads(it.get("meta_json", "{}")).get("chunk_type")) != "parent_rule"
    ]
    neighbors_e = to_evidence(window_wo_anchor[:6], why="Соседние подпункты")

    glossary_e = to_evidence((r.get("defs", [])[:2] + r.get("abbr", [])[:2]), why="Глоссарий")

    refs_e: List[Dict[str, Any]] = []
    try:
        meta_anchor = json.loads(anchor_raw.get("meta_json", "{}"))
        xrefs = meta_anchor.get("cross_references") or []
        if xrefs:
            cand_map = {c["chunk_id"]: c for c in r.get("rules", [])}
            take = [cand_map[x] for x in xrefs if x in cand_map][:2]
            refs_e = to_evidence(take, why="Цитируемые документы/пункты")
    except Exception:
        pass

    evid = pack_evidence(anchor_e, neighbors_e, glossary_e, refs_e)

    # дедуп по cite
    seen = set()
    dedup = []
    for e in evid:
        if e.cite in seen:
            continue
        seen.add(e.cite)
        dedup.append(e)
    evid = dedup

    # нормализуем effective_date у соседей, если «1970-...» и тот же документ
    anchor_date = anchor_e.doc.get("effective_date")
    anchor_doc = anchor_e.doc.get("id")
    if anchor_doc and anchor_date:
        for e in evid:
            d = e.doc.get("effective_date") or ""
            if (not isinstance(d, str)) or d.startswith("1970-"):
                if e.doc.get("id") == anchor_doc:
                    e.doc["effective_date"] = anchor_date

    system = render_system_prompt()
    user = render_user_prompt(req.query, evid, history=hist)
    out_text = await chat(system, user, temperature=0.15, max_tokens=900)
    fallback = f"[{anchor_e.doc['id']} п.{anchor_e.loc.get('chunk_rule_number') or '?'}]"
    out_text = _ensure_citations(out_text, fallback)

    tips = await _suggest_followups(req.query, lang, today, forced_doc_id)

    resp = {"answer": out_text, "evidence": [e.model_dump() for e in evid], "hints": tips[:3]}
    if getattr(req, "conversation_id", None):
        HISTORY[req.conversation_id].append({"role": "user", "content": req.query})
        HISTORY[req.conversation_id].append({"role": "assistant", "content": out_text})
    return resp


