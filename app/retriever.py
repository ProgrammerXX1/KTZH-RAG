import json
import urllib.parse as urlparse
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, date
import numpy as np
from weaviate.classes.query import MetadataQuery
import weaviate

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

_CHUNK_TYPES = (
    "paragraph","parent_rule","list_item","table",
    "definition","abbreviation","appendix","appendix_table","section_title"
)

def chunk_types_overview(limit_per_type: int = 3, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращает срез по коллекциям и chunk_type:
    {
      "Rules": {
        "total": 123,
        "by_type": {
          "paragraph": {
            "count": 100,
            "samples": [ {chunk_id, chunk_type, source_document_id, ...}, ... ]
          },
          ...
        }
      },
      "Definitions": { ... },
      "Abbreviations": { ... }
    }
    Можно сузить по конкретному документу doc_id.
    """
    out: Dict[str, Any] = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if not client.collections.exists(cls):
                out[cls] = {"total": 0, "by_type": {}}
                continue

            coll = client.collections.get(cls)

            # общий total (с учётом doc_id, если задан)
            base_filters = []
            if doc_id:
                base_filters.append(
                    weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id)
                )
            base_filter = (weaviate.classes.query.Filter.all_of(base_filters)
                           if base_filters else None)

            if base_filter is None:
                agg_total = coll.aggregate.over_all(total_count=True)
            else:
                agg_total = coll.aggregate.over_all(total_count=True, filters=base_filter)

            total = int(getattr(agg_total, "total_count", 0))
            by_type: Dict[str, Any] = {}

            for t in _CHUNK_TYPES:
                # where: (base_filters) AND (chunk_type == t)
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

                # пары примеров по типу
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

# Классы
RULES = "Rules"
DEFS = "Definitions"
ABBR = "Abbreviations"

_nat_num = re.compile(r"\d+")


def _client():
    """
    Подключение к Weaviate по URL из env (например, http://weaviate-test:8080).
    """
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
    """
    Создаём коллекции без встроенного векторизатора, со "сплющенными" полями
    и отдельным TEXT-полем meta_json для полного metadata.
    """
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if not client.collections.exists(cls):
                client.collections.create(
                    cls,
                    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
                    properties=[
                        weaviate.classes.config.Property(
                            name="chunk_id",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=True,
                        ),
                        weaviate.classes.config.Property(
                            name="content",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=False,
                            index_searchable=True,
                        ),
                        # плоские поля
                        weaviate.classes.config.Property(
                            name="language",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="chunk_type",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="source_document_id",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="effective_date",
                            data_type=weaviate.classes.config.DataType.DATE,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="section_number",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="chunk_rule_number",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        weaviate.classes.config.Property(
                            name="parent_rule_number",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=True,
                            index_searchable=False,
                        ),
                        # полный metadata как JSON-строка
                        weaviate.classes.config.Property(
                            name="meta_json",
                            data_type=weaviate.classes.config.DataType.TEXT,
                            index_filterable=False,
                            index_searchable=False,
                        ),
                    ],
                )


def reset_schema():
    """
    Полный сброс: дроп коллекций и пересоздание схемы.
    """
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                client.collections.delete(cls)
        # пересоздаём
        ensure_schema()


def wipe_all():
    """
    Очистка всех коллекций (truncate).
    """
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                coll = client.collections.get(cls)
                # удаляем всё без фильтра
                coll.data.delete_many()  # доступно в v4 python-клиенте


def delete_by_source_document_id(doc_id: str) -> Dict[str, Any]:
    """
    Удаляет все объекты с данным source_document_id из всех коллекций.
    Возвращает примерную статистику (если клиент вернул её).
    """
    stats: Dict[str, Any] = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if not client.collections.exists(cls):
                continue
            coll = client.collections.get(cls)
            where = weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id)
            resp = coll.data.delete_many(where=where)
            # resp может быть None (в зависимости от версии). Сохраним как есть.
            stats[cls] = resp if resp is not None else "ok"
    return stats


def collections_counts() -> Dict[str, int]:
    """
    Возвращает количество объектов по коллекциям.
    """
    out = {}
    with _client() as client:
        for cls in (RULES, DEFS, ABBR):
            if client.collections.exists(cls):
                coll = client.collections.get(cls)
                agg = coll.aggregate.over_all(total_count=True)
                out[cls] = int(getattr(agg, "total_count", 0))
            else:
                out[cls] = 0
    return out


def _pick_collection(chunk_type: str) -> str:
    if chunk_type in ("definition",):
        return DEFS
    if chunk_type in ("abbreviation",):
        return ABBR
    return RULES


def _to_rfc3339(date_str: str, end_of_day: bool = False) -> str:
    """
    "2024-11-07" -> "2024-11-07T00:00:00Z" или "...T23:59:59Z"
    """
    if not date_str:
        return "1970-01-01T00:00:00Z"
    if "T" in date_str:
        return date_str
    return f"{date_str}{'T23:59:59Z' if end_of_day else 'T00:00:00Z'}"


async def ingest_chunks(chunks: List[Chunk]) -> int:
    """
    Асинхронная загрузка чанков.
    """
    texts = [f"passage: {c.content}" for c in chunks]
    vecs = await embed_texts(texts)

    with _client() as client:
        count = 0
        for c, v in zip(chunks, vecs):
            coll = client.collections.get(_pick_collection(c.metadata.chunk_type))
            meta = c.metadata.model_dump()
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
                "meta_json": json.dumps(meta, ensure_ascii=False),
            }
            coll.data.insert(properties=props, vector=v.tolist())
            count += 1
    return count


def _natural_key(s: str) -> list:
    # "3.10.2" -> [3, '.', 10, '.', 2] для корректной сортировки
    s = (s or "").lower()
    return [int(text) if text.isdigit() else text for text in _nat_num.split(s) for text in (text,)]


def _parse_date(s) -> datetime:
    """
    Принимает datetime | date | str | None и возвращает naive datetime (UTC-наивный).
    Поддерживает строки 'YYYY-MM-DD' и RFC3339 ('YYYY-MM-DDTHH:MM:SSZ' / +00:00).
    """
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


async def _embed_query(q: str) -> list[float]:
    """Эмбеддинг для текста запроса (одним вызовом, без внешних зависимостей)."""
    vec = await embed_texts([f"query: {q}"], batch_size=1)
    return vec[0].tolist()


def _alpha_for_query(q: str) -> float:
    toks = len([t for t in q.strip().split() if t])
    return HYBRID_ALPHA_SHORT if toks < 5 else HYBRID_ALPHA_LONG


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


async def _hybrid_search(
    collection,
    query: str,
    k: int,
    language: str,
    today_rfc3339: str,
    qv: list[float],
    alpha: float,
    doc_id: str | None = None,   # ← добавили
) -> List[Dict[str, Any]]:
    # 1) Собираем фильтры
    fltrs = [
        weaviate.classes.query.Filter.by_property("language").equal(language),
        weaviate.classes.query.Filter.by_property("effective_date").less_or_equal(today_rfc3339),
    ]
    if doc_id:
        fltrs.append(weaviate.classes.query.Filter.by_property("source_document_id").equal(doc_id))

    # 2) Запрос
    res = collection.query.hybrid(
        query=query,
        vector=qv,
        alpha=alpha,
        limit=max(k, max(64, 3 * MMR_K)),  # запас на MMR/фильтры
        return_properties=[
            "chunk_id",
            "content",
            "meta_json",
            "language",
            "chunk_type",
            "source_document_id",
            "effective_date",
            "section_number",
            "chunk_rule_number",
            "parent_rule_number",
        ],
        filters=weaviate.classes.query.Filter.all_of(fltrs),
        return_metadata=MetadataQuery(score=True),  # ← просим вектор
    )

    # 3) Преобразуем объекты
    rows = []
    for o in res.objects:
        row = o.properties
        row["_score"] = getattr(o.metadata, "score", None)
        # try:
        #     vec = getattr(o.metadata, "vector", None)
        #     row["_vec"] = np.array(vec, dtype=np.float32) if vec is not None else None
        # except Exception:
        #     row["_vec"] = None
        rows.append(row)

    # 4) Оставляем только последнюю редакцию документа (если надо)
    if ONLY_LATEST_REV and rows:
        max_dt: dict[str, datetime] = {}
        for r in rows:
            doc = _doc_id_of(r)
            if not doc:
                continue
            dt = _parse_date(_eff_date_of(r))
            if dt >= max_dt.get(doc, datetime(1970, 1, 1)):
                max_dt[doc] = dt

        filtered = []
        for r in rows:
            doc = _doc_id_of(r)
            if not doc:
                continue
            dt = _parse_date(_eff_date_of(r))
            if dt == max_dt.get(doc, datetime.min):
                filtered.append(r)
        rows = filtered

    return rows


def _aggregate_docs(rows: list[dict]) -> list[tuple[str, float]]:
    # агрегируем до уровня doc_id по max(_score)
    agg: dict[str, float] = {}
    for r in rows:
        doc = _doc_id_of(r)
        if not doc:
            continue
        sc = r.get("_score") or 0.0
        if sc > agg.get(doc, -1e9):
            agg[doc] = sc
    # убывающий топ документов
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


def _mmr_select(cands: list[dict], k: int, lam: float) -> list[dict]:
    """
    MMR на косинусной близости по векторам (_vec), с лёгкой защитой от повторяющихся parent_rule_number.
    lam в [0..1]: 1 — больше новизны, 0 — больше близости к запросу (т.е. score).
    """
    if not cands:
        return []

    # нормализуем векторы (если нет _vec, считаем нулём — тогда работает «как раньше»)
    vecs = []
    for r in cands:
        v = r.get("_vec")
        if isinstance(v, np.ndarray) and v.size:
            n = np.linalg.norm(v)
            vecs.append(v / max(n, 1e-8))
        else:
            vecs.append(None)

    # релевантность заменим на нормированный _score
    scores = np.array([float(r.get("_score") or 0.0) for r in cands], dtype=np.float32)
    if scores.max() > 0:
        scores = scores / scores.max()

    selected = []
    used_prn = set()

    # инициализация — берём лучший по score
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

            # новизна: 1 - max cosine(sim) до уже выбранных
            if vecs[i] is None or all(vecs[j] is None for j in selected):
                novelty = 1.0  # не знаем векторов — не штрафуем
            else:
                sims = []
                for j in selected:
                    if vecs[j] is None:
                        continue
                    sims.append(float(np.dot(vecs[i], vecs[j])))
                novelty = 1.0 - (max(sims) if sims else 0.0)

            mmr = lam * novelty + (1.0 - lam) * float(scores[i])

            # лёгкая защита от повторов одного parent_rule_number
            prn_i = cands[i].get("parent_rule_number") or cands[i].get("section_number") or ""
            if prn_i in used_prn:
                mmr *= 0.85  # маленький штраф

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
    language: str | None = None,
    today: str = "9999-12-31",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Асинхронный поисковый вход — вызывай с await.
    """
    lang = language or DEFAULT_LANG
    today_rfc3339 = _to_rfc3339(today, end_of_day=True)

    # один эмбеддинг запроса на всё
    qvec = await _embed_query(query)
    alpha = _alpha_for_query(query)

    with _client() as client:
        rules_all = await _hybrid_search(client.collections.get(RULES), query, k_rules, lang, today_rfc3339, qvec, alpha)
        defs_all = await _hybrid_search(client.collections.get(DEFS), query, min(k_defs, 30), lang, today_rfc3339, qvec, alpha)
        abbr_all = await _hybrid_search(client.collections.get(ABBR), query, min(k_defs, 30), lang, today_rfc3339, qvec, alpha)

    # --- DOC TOP-N: выбираем 1–3 лучших документов по Rules и режем кандидатов по ним ---
    doc_rank = _aggregate_docs(rules_all)
    keep_docs: set[str]
    if len(doc_rank) >= 2 and doc_rank[0][1] >= doc_rank[1][1] * (1 + MONODOC_MARGIN):
        keep_docs = {doc_rank[0][0]}
    else:
        keep_docs = {d for d, _ in doc_rank[:max(1, DOC_TOPN)]} or {r.get("source_document_id") for r in rules_all[:1]}


    rules = [r for r in rules_all if (_doc_id_of(r) in keep_docs)]
    # defs/abbr теперь ЖЁСТКО только из того же документа, что и якорь (а не просто из keep_docs)
    anchor_doc_id = _doc_id_of(max(rules, key=lambda x: (x.get("_score") or 0.0))) if rules else None
    defs_ = [r for r in defs_all if (_doc_id_of(r) == anchor_doc_id)]
    abbr  = [r for r in abbr_all if (_doc_id_of(r) == anchor_doc_id)]

    # --- MMR по Rules, чтобы не залипать в один подпункт ---
    rules = _mmr_select(rules, k=min(k_rules, MMR_K), lam=MMR_LAMBDA)

    return {"rules": rules, "defs": defs_, "abbr": abbr}


def build_section_window(seed: Dict[str, Any], neighbors: int = 4) -> List[Dict[str, Any]]:
    """
    Окно вокруг пункта по parent_rule_number.
    """
    parent = seed.get("parent_rule_number") or ""
    if not parent:
        return [seed]
    sid = seed.get("source_document_id")
    with _client() as client:
        coll = client.collections.get(RULES)
        res = coll.query.fetch_objects(
            limit=200,
            return_properties=[
                "chunk_id",
                "content",
                "meta_json",
                "language",
                "chunk_type",
                "source_document_id",
                "effective_date",
                "section_number",
                "chunk_rule_number",
                "parent_rule_number",
            ],
            filters=weaviate.classes.query.Filter.all_of(
                [
                    weaviate.classes.query.Filter.by_property("source_document_id").equal(sid),
                    weaviate.classes.query.Filter.by_property("parent_rule_number").equal(parent),
                ]
            ),
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
        # попробовать вставить сам родитель
        parent_rows = []
        for x in all_items:
            if x.get("chunk_rule_number") == parent:
                parent_rows.append(x)
                break
            try:
                mj = json.loads(x.get("meta_json", "{}"))
                if mj.get("chunk_type") == "parent_rule":
                    parent_rows.append(x)
                    break
            except Exception:
                pass
        out = (parent_rows[:1] + out)[: 1 + len(out)]
        return out or [seed]


def to_evidence(items: List[Dict[str, Any]], why: str) -> List[Evidence]:
    """
    Формирование evidence для ответа.
    """
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
