# app/llm.py
import json
import asyncio
import httpx

from .config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


async def _post(payload: dict) -> tuple[int, str]:
    """
    Универсальный POST в /v1/completions.
    Возвращает (status_code, raw_body_text).
    """
    url = f"{LLM_API_BASE}/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(url, headers=headers, json=payload)
        body = (await r.aread()).decode("utf-8", "ignore")
        return r.status_code, body


def _extract_text_from_completion(body: str) -> str:
    """
    Пытаемся аккуратно вытащить текст из ответа модели.
    Поддерживаем оба формата:
      - completions: choices[0].text
      - chat-like:  choices[0].message.content
    """
    try:
        data = json.loads(body)
    except Exception:
        return ""

    try:
        choices = data.get("choices") or []
        if not choices:
            return ""

        ch = choices[0]

        # формат обычного completion
        if isinstance(ch.get("text"), str):
            return ch["text"]

        # формат chat-completion (на будущее)
        msg = ch.get("message") or {}
        if isinstance(msg.get("content"), str):
            return msg["content"]

        return ""
    except Exception:
        return ""


async def chat(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> str:
    """
    вместо /v1/chat/completions используем /v1/completions.

    Собираем prompt так:
        <system>\n\nПОЛЬЗОВАТЕЛЬ:\n<user>\n\nОТВЕТ:
    чтобы модель видела инструкцию и вопрос.
    3 попытки с уменьшением max_tokens.
    """

    system = (system or "").strip()
    user = (user or "").strip()

    base_prompt = f"{system}\n\nПОЛЬЗОВАТЕЛЬ:\n{user}\n\nОТВЕТ:\n"

    # базовый payload
    base = {
        "model": LLM_MODEL,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "prompt": base_prompt,
        "stream": False,
    }

    attempts = [
        base,
        {**base, "max_tokens": min(700, max_tokens)},
        {**base, "max_tokens": min(600, max_tokens)},
    ]

    last_err = "unknown"

    for i, payload in enumerate(attempts, 1):
        try:
            status, body = await _post(payload)
            if 200 <= status < 300:
                text = _extract_text_from_completion(body)
                if not text:
                    last_err = f"empty_text_from_model: {body[:400]}"
                else:
                    return text.strip()
            else:
                last_err = f"LLM HTTP {status}: {body[:800]}"
        except Exception as e:
            last_err = f"LLM exception: {repr(e)}"

        # лёгкий backoff между попытками
        await asyncio.sleep(0.6 * i)

    raise RuntimeError(f"llm_failed: {last_err}")
