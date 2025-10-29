# app/retriever.py
from __future__ import annotations

import json
import re
import sys
import urllib.parse as urlparse
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import numpy as np
import weaviate
from weaviate.classes.query import MetadataQuery

from .config import (
    WEAVIATE_URL,
    DEFAULT_LANG,
    HYBRID_ALPHA_SHORT,
    HYBRID_ALPHA_LONG,
    DOC_TOPN,
    MMR_LAMBDA,
    MMR_K,
    ONLY_LATEST_REV,
    MONODOC_MARGIN,
)
from .embedder import embed_texts
from .models import Chunk, Evidence

# --- коллекции ---
RULES = "Rules"
DEFS = "Definitions"
ABBR = "Abbreviations"

# --- типы чанков (для админ-обзора) ---
_CHUNK_TYPES = (
    "paragraph", "parent_rule", "list_item", "table",
    "definition", "abbreviation", "appendix", "appendix_table", "section_title"
)

_nat_split = re.compile(r"(\d+)")

# --- приоры по документам (мягкий буст) ---
DOC_PRIORS: list[tuple[str, str, float]] = [
    (r"\bву[-\s]?45\b|\bопробовани[ея]\b|\bсокращенн[ао]е\b", "808-2022 ПКБ ЦВ", 2.8),
    (r"\bдиспетчер|\bманевр|\bпри[её]м|\bотправ|\bдц\b|\bаб\b", "291-ИДП", 2.4),
    (r"\bконтактн(ая|ой) сеть|\bнатяжен|\bстрел[аы] провиса|\bркс\b", "1182-ЦЗ", 2.6),
    (r"\bптэ\b|\bправила технической эксплуатации\b", "ПТЭ-544", 2.2),
    (r"\bэ[пп]?т\b|\bавтотормоз|\bkz4at|kz8a|вл80", "854-ЦЗ", 2.0),
]

def _client():
    u = urlparse.urlparse(WEAVIATE_URL)
    host = u.hostname or "weaviate-test"
    port = u.port or 8080
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        grpc_host=host,
        grpc_port=50051,
        http_secure=(u.scheme == "https"),
        grpc_secure=False,
    )

def ensure_schema():
    from weaviate.classes import config as wcc
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                continue
            client.collections.create(
                name=cls,
                vector_config=wcc.Configure.Vector.none(),
                properties=[
                    wcc.Property(name="chunk_id",            data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=True),
                    wcc.Property(name="content",             data_type=wcc.DataType.TEXT, index_filterable=False, index_searchable=True),
                    wcc.Property(name="language",            data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="chunk_type",          data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="source_document_id",  data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="effective_date",      data_type=wcc.DataType.DATE, index_filterable=True, index_searchable=False),
                    wcc.Property(name="section_number",      data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="chunk_rule_number",   data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="parent_rule_number",  data_type=wcc.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wcc.Property(name="meta_json",           data_type=wcc.DataType.TEXT, index_filterable=False, index_searchable=False),
                ],
            )

def reset_schema():
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                client.collections.delete(cls)
    ensure_schema()

def wipe_all():
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                client.collections.get(cls).data.delete_many()

def delete_by_source_document_id(doc_id: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if not client.collections.exists(cls):
                continue
            coll = client.collections.get(cls)
            where = weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id)
            resp = coll.data.delete_many(where=where)
            stats[cls] = resp if resp is not None else "ok"
    return stats

def collections_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                agg = client.collections.get(cls).aggregate.over_all(total_count=True)
                out[cls] = int(getattr(agg, "total_count", 0))
            else:
                out[cls] = 0
    return out

def _pick_collection(chunk_type: str) -> str:
    if chunk_type == "definition":
        return DEFS
    if chunk_type == "abbreviation":
        return ABBR
    return RULES

def _to_rfc3339(date_str: str, end_of_day: bool = False) -> str:
    if not date_str:
        return "1970-01-01T00:00:00Z"
    if "T" in date_str:
        return date_str
    return f"{date_str}{'T23:59:59Z' if end_of_day else 'T00:00:00Z'}"

async def ingest_chunks(chunks: List[Chunk]) -> int:
    texts = [f"passage: {c.content}" for c in chunks]
    vecs = await embed_texts(texts)  # список len(chunks), элементы np.ndarray или None
    ok = [(c, v) for c, v in zip(chunks, vecs) if v is not None]
    bad = [c.chunk_id for c, v in zip(chunks, vecs) if v is None]
    if bad:
        print(f"⚠️ embed skipped {len(bad)} items; first={bad[:3]}", file=sys.stderr)
    if not ok:
        raise RuntimeError("embedder_failed: no vectors")

    with _client() as client:
        count = 0
        for c, v in ok:
            coll = client.collections.get(_pick_collection(c.metadata.chunk_type))
            props = {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "language": c.metadata.language,
                "chunk_type": c.metadata.chunk_type,
                "source_document_id": c.metadata.source_document_id,
                "effective_date": _to_rfc3339(c.metadata.effective_date, end_of_day=True),
                "section_number": c.metadata.section_number or "",
                "chunk_rule_number": c.metadata.chunk_rule_number or "",
                "parent_rule_number": c.metadata.parent_rule_number or "",
                "meta_json": json.dumps(c.metadata.model_dump(), ensure_ascii=False),
            }
            coll.data.insert(properties=props, vector=np.asarray(v, dtype=np.float32).tolist())
            count += 1
    return count

def _natural_key(s: str) -> list:
    s = (s or "").lower()
    parts = _nat_split.split(s)
    return [int(p) if p.isdigit() else p for p in parts]

def _parse_date(s) -> datetime:
    if s is None:
        return datetime(1970, 1, 1)
    if isinstance(s, datetime):
        return s.replace(tzinfo=None)
    if isinstance(s, date):
        return datetime(s.year, s.month, s.day)
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return datetime(1970, 1, 1)
    if "T" in s:
        iso = s.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(iso).replace(tzinfo=None)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime(1970, 1, 1)

# --- безопасный эмбеддинг запроса с фолбэком ---
def _clean_query(q: str) -> str:
    q = (q or "").replace("\x00", " ").strip()
    return q[:2000]

async def _embed_query_optional(q: str) -> Optional[List[float]]:
    q = _clean_query(q)
    vecs = await embed_texts([f"query: {q}"], batch_size=1)
    v = vecs[0] if isinstance(vecs, list) else vecs[0]
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).tolist()

def _alpha_for_query(q: str, has_vec: bool) -> float:
    if not has_vec:
        return 0.0
    toks = len([t for t in q.strip().split() if t])
    has_number = bool(re.search(r"\d", q))
    base = HYBRID_ALPHA_SHORT if toks < 5 else HYBRID_ALPHA_LONG
    return max(0.45, base - 0.06) if has_number else base

def _doc_id_of(r: dict) -> str:
    try:
        return r.get("source_document_id") or json.loads(r.get("meta_json", "{}")).get("source_document_id") or ""
    except Exception:
        return r.get("source_document_id") or ""

def _eff_date_of(r: dict):
    try:
        return r.get("effective_date") or json.loads(r.get("meta_json", "{}")).get("effective_date")
    except Exception:
        return r.get("effective_date")

async def _attach_vectors_locally(rows: List[Dict[str, Any]], top_n: int = 256) -> None:
    if not rows:
        return
    sorted_idx = sorted(range(len(rows)), key=lambda i: float(rows[i].get("_score") or 0.0), reverse=True)
    take_idx = sorted_idx[: min(top_n, len(rows))]
    payload = [f"passage: {(rows[i].get('content') or '').strip()}" for i in take_idx]
    if not any(t.strip() for t in payload):
        for i in range(len(rows)):
            rows[i]["_vec"] = None
        return
    vecs = await embed_texts(payload, batch_size=32)

    if isinstance(vecs, np.ndarray):
        m = min(vecs.shape[0], len(take_idx))
        get_vec = lambda j: vecs[j, :]
    else:
        m = min(len(vecs), len(take_idx))
        get_vec = lambda j: vecs[j]

    for j in range(m):
        i = take_idx[j]
        v = get_vec(j)
        try:
            v = np.asarray(v, dtype=np.float32)
        except Exception:
            v = None
        rows[i]["_vec"] = v
    for j in range(m, len(take_idx)):
        rows[take_idx[j]]["_vec"] = None
    for i in sorted_idx[min(top_n, len(rows)):]:
        rows[i]["_vec"] = None

# --- поиск с фолбэком BM25 ---
async def _search_collection(
    collection,
    query: str,
    k: int,
    language: str,
    today_rfc3339: str,
    qv: Optional[List[float]],
    alpha: float,
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fltrs = [
        weaviate.classes.query.Filter.by_property("language").equal(language),
        weaviate.classes.query.Filter.by_property("effective_date").less_or_equal(today_rfc3339),
    ]
    if doc_id:
        fltrs.append(weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id))
    where = weaviate.classes.query.Filter.all_of(fltrs)

    limit = max(k, max(64, 3 * MMR_K))
    ret_props = [
        "chunk_id", "content", "meta_json", "language", "chunk_type",
        "source_document_id", "effective_date", "section_number",
        "chunk_rule_number", "parent_rule_number",
    ]

    if qv is not None:
        res = collection.query.hybrid(
            query=query,
            vector=qv,
            alpha=alpha,
            limit=limit,
            return_properties=ret_props,
            filters=where,
            return_metadata=MetadataQuery(score=True),
        )
    else:
        res = collection.query.bm25(
            query=query,
            limit=limit,
            return_properties=ret_props,
            filters=where,
            return_metadata=MetadataQuery(score=True),
        )

    rows: List[Dict[str, Any]] = []
    for o in res.objects:
        row = o.properties
        row["_score"] = getattr(o.metadata, "score", None)
        row["_vec"] = None
        rows.append(row)

    if ONLY_LATEST_REV and rows:
        max_dt: Dict[str, datetime] = {}
        for r in rows:
            doc = _doc_id_of(r)
            if not doc:
                continue
            dt = _parse_date(_eff_date_of(r))
            if dt >= max_dt.get(doc, datetime(1970, 1, 1)):
                max_dt[doc] = dt
        rows = [r for r in rows if _doc_id_of(r) and _parse_date(_eff_date_of(r)) == max_dt[_doc_id_of(r)]]

    await _attach_vectors_locally(rows, top_n=max(256, 3 * MMR_K))
    return rows

def _aggregate_docs(rows: List[dict]) -> List[tuple[str, float]]:
    agg: Dict[str, float] = {}
    for r in rows:
        doc = _doc_id_of(r)
        if not doc:
            continue
        sc = float(r.get("_score") or 0.0)
        if sc > agg.get(doc, -1e9):
            agg[doc] = sc
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)

def _apply_doc_priors(query: str, rows: List[dict]) -> Dict[str, float]:
    q = query.lower()
    prior: Dict[str, float] = {}
    for pat, doc, w in DOC_PRIORS:
        if re.search(pat, q):
            prior[doc] = prior.get(doc, 0.0) + w
    if not prior:
        return {}
    for r in rows:
        d = _doc_id_of(r)
        if d in prior:
            r["_score"] = float(r.get("_score") or 0.0) + prior[d]
    return prior

def _boost_chunk_number_if_present(query: str, rows: List[dict]) -> None:
    m = re.search(r"(?:п\.?|пункт)\s*(\d+(\.\d+)*)\b", (query or "").lower())
    if not m:
        return
    want = m.group(1)
    for r in rows:
        crn = (r.get("chunk_rule_number") or "").strip(".")
        if crn == want or crn.startswith(want + "."):
            r["_score"] = float(r.get("_score") or 0.0) + 8.0

def _mmr_select(cands: List[dict], k: int, lam: float) -> List[dict]:
    if not cands:
        return []
    scores = np.array([float(r.get("_score") or 0.0) for r in cands], dtype=np.float32)
    if scores.max() > 0:
        scores = scores / scores.max()

    vecs = []
    for r in cands:
        v = r.get("_vec")
        if isinstance(v, np.ndarray) and v.size:
            n = float(np.linalg.norm(v))
            vecs.append(v / max(n, 1e-8))
        else:
            vecs.append(None)

    selected: List[int] = []
    used_prn = set()

    first = int(np.argmax(scores))
    selected.append(first)
    prn = cands[first].get("parent_rule_number") or cands[first].get("section_number") or ""
    if prn:
        used_prn.add(prn)

    while len(selected) < min(k, len(cands)):
        best_idx = None
        best_val = -1e9
        for i in range(len(cands)):
            if i in selected:
                continue
            prn_i = cands[i].get("parent_rule_number") or cands[i].get("section_number") or ""
            if vecs[i] is None or all(vecs[j] is None for j in selected):
                novelty = 1.0
            else:
                sims = [float(np.dot(vecs[i], vecs[j])) for j in selected if vecs[j] is not None]
                novelty = 1.0 - (max(sims) if sims else 0.0)
            if prn_i in used_prn:
                novelty *= 0.95
            mmr = lam * novelty + (1.0 - lam) * float(scores[i])
            if mmr > best_val:
                best_val = mmr
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        prn = cands[best_idx].get("parent_rule_number") or cands[best_idx].get("section_number") or ""
        if prn:
            used_prn.add(prn)

    return [cands[i] for i in selected]

async def search(
    query: str,
    k_rules: int = 100,
    k_defs: int = 60,
    language: Optional[str] = None,
    today: str = "9999-12-31",
    doc_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    lang = language or DEFAULT_LANG
    today_rfc3339 = _to_rfc3339(today, end_of_day=True)

    qvec = await _embed_query_optional(query)  # может быть None
    alpha = _alpha_for_query(query, has_vec=(qvec is not None))
    if qvec is None:
        print("⚠️ query_embedding=None → BM25 fallback", file=sys.stderr)

    q_low = query.lower()
    is_define = any(p in q_low for p in (
        "что означает", "что такое", "дайте определение", "определение",
        "расшифровка", "что означает сокращение", "что означает термин"
    ))

    with _client() as client:
        if is_define:
            defs_all  = await _search_collection(client.collections.get(DEFS),  query, max(30, k_defs), lang, today_rfc3339, qvec, alpha, doc_id)
            abbr_all  = await _search_collection(client.collections.get(ABBR),  query, max(30, k_defs), lang, today_rfc3339, qvec, alpha, doc_id)
            rules_all = await _search_collection(client.collections.get(RULES), query, min(40, k_rules), lang, today_rfc3339, qvec, alpha, doc_id)
        else:
            rules_all = await _search_collection(client.collections.get(RULES), query, k_rules,        lang, today_rfc3339, qvec, alpha, doc_id)
            defs_all  = await _search_collection(client.collections.get(DEFS),  query, min(k_defs,30), lang, today_rfc3339, qvec, alpha, doc_id)
            abbr_all  = await _search_collection(client.collections.get(ABBR),  query, min(k_defs,30), lang, today_rfc3339, qvec, alpha, doc_id)

    # точечные бусты до агрегации
    _boost_chunk_number_if_present(query, rules_all)
    prior = _apply_doc_priors(query, rules_all)

    doc_rank = _aggregate_docs(rules_all)
    if not doc_rank:
        return {"rules": [], "defs": [], "abbr": []}

    # выбор корпуса документов (форс-монодок при явном номере пункта)
    has_point = bool(re.search(r"\b(\d+(?:\.\d+)*)\b", query))
    if has_point:
        keep_docs: set[str] = {doc_rank[0][0]}
    elif len(doc_rank) >= 2 and doc_rank[0][1] >= doc_rank[1][1] * (1 + MONODOC_MARGIN):
        keep_docs = {doc_rank[0][0]}
    else:
        if prior:
            top_prior_doc = max(prior.items(), key=lambda x: x[1])[0]
            keep_docs = {top_prior_doc}
        else:
            keep_docs = {d for d, _ in doc_rank[:max(1, DOC_TOPN)]} or {_doc_id_of(rules_all[0])}

    rules = [r for r in rules_all if (_doc_id_of(r) in keep_docs)]

    anchor_doc_id = _doc_id_of(max(rules, key=lambda x: (x.get("_score") or 0.0))) if rules else None
    if is_define and anchor_doc_id is None and defs_all:
        anchor_doc_id = _doc_id_of(defs_all[0])

    defs_ = [r for r in defs_all if (anchor_doc_id is None or _doc_id_of(r) == anchor_doc_id)]
    abbr = [r for r in abbr_all if (anchor_doc_id is None or _doc_id_of(r) == anchor_doc_id)]

    # финальный отбор rule-кандидатов MMR
    rules = _mmr_select(rules, k=min(k_rules, MMR_K), lam=MMR_LAMBDA)

    return {"rules": rules, "defs": defs_, "abbr": abbr}

def build_section_window(seed: Dict[str, Any], neighbors: int = 10) -> List[Dict[str, Any]]:
    parent = seed.get("parent_rule_number") or ""
    if not parent:
        return [seed]
    sid = seed.get("source_document_id")
    with _client() as client:
        coll = client.collections.get(RULES)
        res = coll.query.fetch_objects(
            limit=200,
            return_properties=[
                "chunk_id", "content", "meta_json", "language", "chunk_type",
                "source_document_id", "effective_date", "section_number",
                "chunk_rule_number", "parent_rule_number",
            ],
            filters=weaviate.classes.query.Filter.all_of([
                weaviate.classes.query.Filter.by_property("source_document_id").equal(sid),
                weaviate.classes.query.Filter.by_property("parent_rule_number").equal(parent),
            ]),
        )
        all_items = sorted(
            [o.properties for o in res.objects],
            key=lambda x: _natural_key(
                x.get("chunk_rule_number") or x.get("section_number") or x.get("chunk_id") or ""
            ),
        )
        try:
            idx = next(i for i, x in enumerate(all_items) if x["chunk_id"] == seed["chunk_id"])
        except StopIteration:
            return [seed]

        lo, hi = max(0, idx - neighbors), min(len(all_items), idx + neighbors + 1)
        out = all_items[lo:hi]

        parent_rows = []
        for it in all_items:
            if it.get("chunk_rule_number") == parent:
                parent_rows.append(it)
                break
            try:
                mj = json.loads(it.get("meta_json", "{}"))
                if mj.get("chunk_type") == "parent_rule" and (mj.get("chunk_rule_number") == parent):
                    parent_rows.append(it)
                    break
            except Exception:
                pass

        out = (parent_rows[:1] + out)[: 1 + len(out)]
        return out or [seed]

def to_evidence(items: List[Dict[str, Any]], why: str) -> List[Evidence]:
    ev: List[Evidence] = []
    for it in items:
        try:
            m = json.loads(it.get("meta_json", "{}"))
        except Exception:
            m = {}
        text = it.get("content", "")
        short = text if len(text) < 600 else text[:600]
        ev.append(
            Evidence(
                cite=it["chunk_id"],
                doc={
                    "id": m.get("source_document_id") or it.get("source_document_id"),
                    "title": m.get("document_title"),
                    "version": m.get("document_version"),
                    "effective_date": m.get("effective_date") or it.get("effective_date"),
                },
                loc={
                    "section_number": m.get("section_number") or it.get("section_number"),
                    "chunk_rule_number": m.get("chunk_rule_number") or it.get("chunk_rule_number"),
                    "page_number": m.get("page_number"),
                    "chunk_type": m.get("chunk_type") or it.get("chunk_type"),
                },
                text=short,
                why=why,
            )
        )
    return ev

def chunk_types_overview(limit_per_type: int = 3, doc_id: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if not client.collections.exists(cls):
                out[cls] = {"total": 0, "by_type": {}}
                continue

            coll = client.collections.get(cls)

            base_filters = []
            if doc_id:
                base_filters.append(weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id))
            base_filter = weaviate.classes.query.Filter.all_of(base_filters) if base_filters else None

            agg_total = coll.aggregate.over_all(total_count=True, filters=base_filter) if base_filter else coll.aggregate.over_all(total_count=True)
            total = int(getattr(agg_total, "total_count", 0))
            by_type: Dict[str, Any] = {}

            for t in _CHUNK_TYPES:
                fltrs = [weaviate.classes.query.Filter.by_property("chunk_type").equal(t)]
                if base_filter is not None:
                    fltrs.insert(0, base_filter)
                    where = weaviate.classes.query.Filter.all_of(fltrs)
                else:
                    where = fltrs[0]

                agg_t = coll.aggregate.over_all(total_count=True, filters=where)
                cnt = int(getattr(agg_t, "total_count", 0))
                if cnt <= 0:
                    continue

                res = coll.query.fetch_objects(
                    limit=max(0, int(limit_per_type)),
                    return_properties=[
                        "chunk_id", "chunk_type", "source_document_id",
                        "section_number", "chunk_rule_number", "parent_rule_number",
                        "effective_date"
                    ],
                    filters=where,
                )
                samples = [o.properties for o in res.objects]
                by_type[t] = {"count": cnt, "samples": samples}

            out[cls] = {"total": total, "by_type": by_type}
    return out
