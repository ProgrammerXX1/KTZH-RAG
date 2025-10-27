import numpy as np
import httpx, time, math, sys
from .config import EMBEDDER_API_BASE, EMBEDDER_API_KEY, EMBEDDER_MODEL

_query_cache: dict[str, np.ndarray] = {}

async def embed_query(text: str) -> np.ndarray:
    # E5-формат запроса уже с префиксом
    key = f"q::{text}"
    if key in _query_cache:
        return _query_cache[key]
    vec = (await embed_texts([f"query: {text}"]))[0]
    _query_cache[key] = vec
    return vec
# app/embedder.py
async def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    total = len(texts)
    all_vecs = []
    errors = []
    start = time.time()
    print(f"\n🚀 embedding {total} chunks via {EMBEDDER_MODEL}...\n", flush=True)

    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            payload = {"model": EMBEDDER_MODEL, "input": batch}
            try:
                r = await client.post(
                    f"{EMBEDDER_API_BASE}/embeddings",
                    headers={
                        "Authorization": f"Bearer {EMBEDDER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                r.raise_for_status()
                body = r.json()
                data = body.get("data") or []
                if len(data) != len(batch):
                    raise RuntimeError(f"embeddings count mismatch: got {len(data)} for batch {len(batch)}; body={str(body)[:400]}")
                vecs = [np.array(d["embedding"], dtype=np.float32) for d in data]
                all_vecs.extend(vecs)
            except Exception as e:
                # копим ошибки и падаем после цикла
                errors.append(f"batch {i//batch_size+1}: {repr(e)}")

            done = min(i+batch_size, total)
            pct = done / total * 100
            sys.stdout.write(f"\r[{done}/{total}] {pct:.1f}%"); sys.stdout.flush()

    dur = time.time() - start
    print(f"\n✅ done {len(all_vecs)}/{total} chunks in {dur:.1f}s\n", flush=True)

    if errors:
        # если ничего не сэмбеддили — валим с детальной ошибкой
        if not all_vecs:
            raise RuntimeError("embedder_failed: no vectors returned;\n" + "\n".join(errors))
        # иначе предупредим, но вернём частичный результат (редко)
        print("⚠️ embedder had errors:\n" + "\n".join(errors), file=sys.stderr)

    if not all_vecs:
        # на всякий случай — дублирующая защита
        raise RuntimeError("embedder_failed: empty result (no batches succeeded)")

    return np.vstack(all_vecs)
