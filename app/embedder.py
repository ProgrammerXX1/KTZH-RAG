# app/embedder.py
from __future__ import annotations

import sys, asyncio
import numpy as np
import httpx
from .config import EMBEDDER_API_BASE, EMBEDDER_API_KEY, EMBEDDER_MODEL

# Кэш для запросных эмбеддингов (экономит latency при похожих вопросах)
_query_cache: dict[str, np.ndarray] = {}

MAX_CHARS_PER_ITEM = 1600  # ~512 токенов

def _truncate(s: str, lim: int = MAX_CHARS_PER_ITEM) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = s.replace("\x00", " ").strip()
    return s[:lim] if len(s) > lim else s

async def embed_texts(texts: list[str], batch_size: int = 32) -> list[np.ndarray | None]:
    """
    Возвращает список эмбеддингов (np.ndarray | None) под каждый входной текст.
    При ошибках/400 батч режется; для упрямых элементов делается повтор с укорочением.
    """
    texts = [_truncate(t) for t in texts]
    total = len(texts)
    out: list[np.ndarray | None] = [None] * total
    print(f"\n🚀 embedding {total} items via {EMBEDDER_MODEL}...\n", flush=True)

    async with httpx.AsyncClient(timeout=120) as client:
        i = 0
        cur_bs = max(1, batch_size)
        while i < total:
            lo, hi = i, min(i + cur_bs, total)
            batch = texts[lo:hi]
            try:
                r = await client.post(
                    f"{EMBEDDER_API_BASE}/embeddings",
                    headers={
                        "Authorization": f"Bearer {EMBEDDER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": EMBEDDER_MODEL, "input": batch},
                )
                if r.status_code == 400 and (hi - lo) > 1:
                    # резать батч вдвое
                    cur_bs = max(1, cur_bs // 2)
                    continue
                r.raise_for_status()
                data = (r.json().get("data") or [])
                if len(data) != len(batch):
                    raise RuntimeError("shape_mismatch")
                for j, d in enumerate(data):
                    out[lo + j] = np.array(d["embedding"], dtype=np.float32)
                i = hi
                # плавно увеличиваем обратно, если всё ок
                if cur_bs < batch_size:
                    cur_bs = min(batch_size, cur_bs * 2)
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 400 and (hi - lo) == 1:
                    # одиночный элемент — ещё сильнее укоротим и попробуем разок
                    t2 = _truncate(batch[0], lim=MAX_CHARS_PER_ITEM // 2)
                    try:
                        r2 = await client.post(
                            f"{EMBEDDER_API_BASE}/embeddings",
                            headers={
                                "Authorization": f"Bearer {EMBEDDER_API_KEY}",
                                "Content-Type": "application/json",
                            },
                            json={"model": EMBEDDER_MODEL, "input": [t2]},
                        )
                        if 200 <= r2.status_code < 300:
                            out[lo] = np.array(r2.json()["data"][0]["embedding"], dtype=np.float32)
                        else:
                            sys.stderr.write(f"\n⚠️ drop idx={lo} due to 400\n")
                    except Exception:
                        sys.stderr.write(f"\n⚠️ drop idx={lo} due to error\n")
                    i = hi
                else:
                    await asyncio.sleep(0.4)
            except Exception:
                await asyncio.sleep(0.4)

    done = sum(1 for v in out if v is not None)
    print(f"\n✅ done {done}/{total} items\n", flush=True)
    return out

async def embed_query(text: str) -> np.ndarray:
    """
    Эмбеддинг запроса (E5 требует префикс 'query: ').
    Кэшируется в памяти.
    """
    t = _truncate(text, lim=1024)
    key = f"q::{t}"
    if key in _query_cache:
        return _query_cache[key]
    vec = (await embed_texts([f"query: {t}"], batch_size=1))[0]
    if vec is None:
        raise RuntimeError("embed_query_failed")
    _query_cache[key] = vec
    return vec
