from __future__ import annotations

import datetime
import io
import json
import zipfile
from typing import Any, Dict, Iterator, List, Optional

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
)
from .parser.pars import router as parser_router


# --- helpers: корректная кириллица в именах внутри ZIP ---
UTF8_FLAG = 0x800  # general purpose bit 11 — "Language encoding flag (EFS)"


def _guess_cyr_name(name: str) -> str:
    """
    zipfile, если нет UTF-8 флага, декодирует filename как cp437.
    Пробуем восстановить: cp437->bytes->cp866/1251.
    """
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
    # Если в заголовке выставлен UTF-8 флаг — имя уже корректное
    if info.flag_bits & UTF8_FLAG:
        return info.filename
    # Иначе пытаемся восстановить кириллицу
    return _guess_cyr_name(info.filename)


def _iter_jsonl_chunks(f: io.TextIOBase) -> Iterator[Chunk]:
    """
    Простой синхронный генератор: читаем построчно JSONL и валидируем через Pydantic.
    Не держим весь файл в памяти.
    """
    for i, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            continue
        try:
            yield Chunk.model_validate_json(s)
        except Exception:
            # тут можно логировать "плохие" строки
            continue


async def _ingest_streaming(file_like: io.TextIOBase, batch_size: int = 1000) -> int:
    """
    Копим чанки батчами и вызываем ingest_chunks.
    """
    buf: list[Chunk] = []
    total = 0
    for ch in _iter_jsonl_chunks(file_like):
        buf.append(ch)
        if len(buf) >= batch_size:
            total += await ingest_chunks(buf)
            buf.clear()
    if buf:
        total += await ingest_chunks(buf)
    return total


# === FastAPI app ===
app = FastAPI(title="KTZH RAG")


@app.on_event("startup")
def startup() -> None:
    ensure_schema()


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


# ---------- Админ/поддержка ----------

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
    """
    Обзор типов чанков в векторном хранилище.
    Параметры:
      - limit_per_type: сколько примеров вернуть на каждый chunk_type (по каждой коллекции)
      - doc_id: опционально сузить статистику по конкретному документу (source_document_id)
    """
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
    per_file: list[dict[str, Any]] = []

    for info in z.infolist():
        # нормализуем имя (кириллица, nested/dir)
        safe_name = _fix_zip_name(info)

        # пропускаем директории
        if info.is_dir():
            continue
        # берём только *.jsonl (учитываем вложенные пути)
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
    results: list[dict[str, Any]] = []
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
    r = await search(req.query, req.k_rules, req.k_defs, language=req.language, today=today, doc_id=req.doc_id)

    # Кандидаты на якорь из Rules
    candidates = (r.get("rules") or [])

    if not candidates:
        defs = r.get("defs", []) or []
        abbr = r.get("abbr", []) or []
        pool = defs[:5] + abbr[:5]
        from collections import Counter

        ids = [
            (json.loads(x.get("meta_json", "{}")).get("source_document_id") or x.get("source_document_id"))
            for x in pool
        ]
        if ids:
            top_id, _ = Counter([i for i in ids if i]).most_common(1)[0]
            pool = [
                x
                for x in pool
                if (json.loads(x.get("meta_json", "{}")).get("source_document_id") or x.get("source_document_id"))
                == top_id
            ]
        glossary_e = to_evidence(pool[:6], why="Глоссарий")
        user = render_user_prompt(req.query, glossary_e)
        out = await chat(render_system_prompt(), user)
        return {"answer": out, "evidence": [e.model_dump() for e in glossary_e]}

    # Берём лучший по _score
    anchor_raw = max(candidates, key=lambda x: (x.get("_score") or 0.0))
    anchor_e = to_evidence([anchor_raw], why="Якорная норма")[0]

    # Окно раздела вокруг якоря
    window_raw = build_section_window(anchor_raw, neighbors=4)
    # убрать дубликат якоря:
    window_wo_anchor = [it for it in window_raw if it.get("chunk_id") != anchor_raw.get("chunk_id")]
    neighbors_e = to_evidence(window_wo_anchor[:6], why="Соседние подпункты")

    # Глоссарий (defs/abbr)
    glossary_e = to_evidence((r.get("defs", [])[:2] + r.get("abbr", [])[:2]), why="Глоссарий")

    # Cross-references из meta_json у якоря (если есть)
    refs_e: list = []
    try:
        meta_anchor = json.loads(anchor_raw.get("meta_json", "{}"))
        xrefs = meta_anchor.get("cross_references") or []
        if xrefs:
            cand_map = {c["chunk_id"]: c for c in r.get("rules", [])}
            take = [cand_map[x] for x in xrefs if x in cand_map][:2]
            refs_e = to_evidence(take, why="Цитируемые документы/пункты")
    except Exception:
        pass

    # Упаковка + дедуп по cite
    evid = pack_evidence(anchor_e, neighbors_e, glossary_e, refs_e)
    seen = set()
    dedup = []
    for e in evid:
        if e.cite in seen:
            continue
        seen.add(e.cite)
        dedup.append(e)
    evid = dedup

    # Ответ
    system = render_system_prompt()
    user = render_user_prompt(req.query, evid)
    out = await chat(system, user)
    return {"answer": out, "evidence": [e.model_dump() for e in evid]}


app.include_router(parser_router)