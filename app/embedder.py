import asyncio
import numpy as np
import httpx, time, sys, random
from .config import EMBEDDER_API_BASE, EMBEDDER_API_KEY, EMBEDDER_MODEL

_query_cache: dict[str, np.ndarray] = {}

def _norm_text(s: str) -> str:
    # лёгкая нормализация, чтобы стабилизировать кэш и эмбеддинги
    return " ".join((s or "").split())

async def embed_query(text: str) -> np.ndarray:
    key = f"q::{_norm_text(text)}::{EMBEDDER_MODEL}"
    if key in _query_cache:
        return _query_cache[key]
    vec = (await embed_texts([f"query: {_norm_text(text)}"], batch_size=1))[0]
    _query_cache[key] = vec
    return vec

async def _post_embeddings(client: httpx.AsyncClient, payload: dict, attempt: int) -> dict:
    r = await client.post(
        f"{EMBEDDER_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {EMBEDDER_API_KEY}", "Content-Type": "application/json"},
        json=payload,
    )
    # 429/5xx считаем ретраибельными
    if r.status_code in (429, 500, 502, 503, 504):
        raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
    r.raise_for_status()
    return r.json()

async def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    total = len(texts)
    all_vecs: list[np.ndarray] = []
    errors = []
    start = time.time()
    print(f"\nembedding {total} chunks via {EMBEDDER_MODEL}\n", flush=True)

    # защита от пустого ввода
    if total == 0:
        return np.zeros((0, 1), dtype=np.float32)

    # ограничим batch_size разумно
    bsz = max(1, min(batch_size, 128))

    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, total, bsz):
            batch = [_norm_text(t) for t in texts[i:i+bsz]]
            payload = {"model": EMBEDDER_MODEL, "input": batch}

            # до 3 попыток с экспоненциальным бэкоффом и джиттером
            ok = False
            last_err = None
            for attempt in range(1, 4):
                try:
                    body = await _post_embeddings(client, payload, attempt)
                    data = body.get("data") or []
                    if len(data) != len(batch):
                        raise RuntimeError(
                            f"embeddings count mismatch: got {len(data)} for batch {len(batch)}; body={str(body)[:400]}"
                        )
                    vecs = [np.array(d["embedding"], dtype=np.float32) for d in data]
                    all_vecs.extend(vecs)
                    ok = True
                    break
                except Exception as e:
                    last_err = e
                    sleep_s = (0.4 * (2 ** (attempt - 1))) + random.uniform(0, 0.2)
                    await asyncio.sleep(sleep_s)

            if not ok:
                errors.append(f"batch {i//bsz+1}: {repr(last_err)}")

            done = min(i + bsz, total)
            pct = done / total * 100
            sys.stdout.write(f"\r[{done}/{total}] {pct:.1f}%")
            sys.stdout.flush()

    dur = time.time() - start
    print(f"\nfinished {len(all_vecs)}/{total} in {dur:.1f}s\n", flush=True)

    if errors and not all_vecs:
        raise RuntimeError("embedder_failed: no vectors returned;\n" + "\n".join(errors))

    if not all_vecs:
        raise RuntimeError("embedder_failed: empty result")

    # склеиваем по порядку
    return np.vstack(all_vecs)
