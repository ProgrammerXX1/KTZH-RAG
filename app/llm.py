# app/llm.py
import json, asyncio, httpx
from .config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL

async def _post(payload: dict) -> tuple[int, str]:
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
        r = await client.post(url, headers=headers, json=payload)
        return r.status_code, (await r.aread()).decode("utf-8", "ignore")

async def chat(system: str, user: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    """
    Жёсткий на точность режим: низкая температура и ограничение длины.
    3 попытки с деградацией max_tokens.
    """
    base = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": None,
    }
    attempts = [
        {**base, "messages":[{"role":"system","content":system},{"role":"user","content":user}]},
        {**base, "max_tokens": min(700, max_tokens), "messages":[{"role":"system","content":system},{"role":"user","content":user}]},
        {**base, "max_tokens": min(600, max_tokens), "messages":[{"role":"user","content":f"{system}\n\n{user}"}]},
    ]
    last = "unknown"
    for i, p in enumerate(attempts, 1):
        try:
            st, body = await _post(p)
            if 200 <= st < 300:
                data = json.loads(body)
                return data["choices"][0]["message"]["content"]
            last = f"LLM HTTP {st}: {body[:800]}"
        except Exception as e:
            last = f"LLM exception: {repr(e)}"
        await asyncio.sleep(0.6 * i)
    raise RuntimeError(f"llm_failed: {last}")
