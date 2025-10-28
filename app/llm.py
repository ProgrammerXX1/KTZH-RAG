import os
import httpx, asyncio, json


LLM_API_BASE = os.getenv("LLM_API_BASE", "http://92.46.59.74:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODEL   = os.getenv("LLM_MODEL", "qwen3-next-80b-a3b")

async def _post(payload: dict) -> dict:
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type":"application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, headers=headers, json=payload)
        # не бросаем сразу — отдадим тело для диагностики
        return {"status": resp.status_code, "json": (await resp.aread()).decode("utf-8", "ignore")}

async def chat(system: str, user: str, temperature: float=0.2, max_tokens: int=800) -> str:
    """
    Делает до 3 попыток с деградацией параметров и упрощением payload.
    """
    base_payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role":"system", "content": system},
            {"role":"user",   "content": user}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    attempts = [
        base_payload,
        {**base_payload, "max_tokens": min(512, max_tokens)},
        # На третьей попытке убираем system и объединяем — иногда серверы так оживают
        {
            "model": LLM_MODEL,
            "messages": [{"role":"user", "content": f"{system}\n\n{user}"}],
            "temperature": temperature,
            "max_tokens": min(384, max_tokens)
        },
    ]

    last_error = None
    for i, payload in enumerate(attempts, 1):
        try:
            res = await _post(payload)
            status = res["status"]
            body   = res["json"]
            if 200 <= status < 300:
                # распарсим JSON вручную, т.к. мы читали сырой текст
                import json
                data = json.loads(body)
                return data["choices"][0]["message"]["content"]
            else:
                last_error = f"LLM HTTP {status}: {body[:800]}"
                # Небольшая пауза перед ретраем — на случай «прогрева»
                await asyncio.sleep(0.8)
        except Exception as e:
            last_error = f"LLM exception: {repr(e)}"
            await asyncio.sleep(0.8)

    raise RuntimeError(f"llm_failed: {last_error}")
