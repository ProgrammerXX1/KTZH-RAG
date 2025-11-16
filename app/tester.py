#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import textwrap

try:
    import requests
except ImportError:
    raise SystemExit("Нужно установить requests: pip install requests")

BASE_URL = "http://92.46.59.74:8000/v1"
API_KEY = "local"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def pretty(obj, limit=800):
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    if len(s) > limit:
        return s[:limit] + "...(truncated)"
    return s


def do_post(name: str, path: str, payload: dict):
    url = BASE_URL + path
    print("=" * 80)
    print(f"TEST: {name}")
    print(f"URL:  {url}")
    print("Payload:")
    print(textwrap.indent(pretty(payload), "  "))
    try:
        resp = requests.post(url, headers=HEADERS, data=json.dumps(payload))
    except Exception as e:
        print(f"EXCEPTION: {e!r}")
        return
    print(f"Status: {resp.status_code}")
    raw = resp.text
    print("Raw response (first 800 chars):")
    print(textwrap.indent(raw[:800], "  "))
    try:
        j = resp.json()
        print("Parsed JSON:")
        print(textwrap.indent(pretty(j), "  "))
    except Exception:
        print("Parsed JSON: <cannot parse>")


def main():
    # 0) /v1/models
    print("=" * 80)
    print("TEST: /v1/models")
    try:
        r = requests.get(BASE_URL + "/models", headers={"Authorization": f"Bearer {API_KEY}"})
        print(f"Status: {r.status_code}")
        print("Raw response:")
        print(textwrap.indent(r.text[:800], "  "))
        try:
            j = r.json()
            print("Parsed JSON:")
            print(textwrap.indent(pretty(j), "  "))
        except Exception:
            print("Parsed JSON: <cannot parse>")
    except Exception as e:
        print(f"EXCEPTION /v1/models: {e!r}")

    # Набор вариантов для /v1/chat/completions
    tests = []

    # 1) Минимальный chat: только user
    tests.append(
        (
            "chat_minimal_user_only",
            "/chat/completions",
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": "ping"}
                ],
            },
        )
    )

    # 2) chat с stream/temp/max_tokens
    tests.append(
        (
            "chat_with_stream_temp_max",
            "/chat/completions",
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": "ping"},
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 16,
            },
        )
    )

    # 3) chat c system + user
    tests.append(
        (
            "chat_with_system_and_user",
            "/chat/completions",
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "ping"},
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 16,
            },
        )
    )

    # 4) chat без model (пусть gateway подставит DEFAULT_MODEL)
    tests.append(
        (
            "chat_without_model",
            "/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "ping without explicit model"},
                ],
                "stream": False,
                "temperature": 0.2,
                "max_tokens": 16,
            },
        )
    )

    # 5) chat с лишним полем stop (extra="allow")
    tests.append(
        (
            "chat_with_stop_field",
            "/chat/completions",
            {
                "model": "gpt-oss-120b",
                "messages": [
                    {"role": "user", "content": "ping with stop"},
                ],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 16,
                "stop": ["\n"],  # лишнее поле
            },
        )
    )

    # 6) legacy /v1/completions (если вдруг chat ломается, а completions жив)
    tests.append(
        (
            "legacy_completions",
            "/completions",
            {
                "model": "gpt-oss-120b",
                "prompt": "ping via /v1/completions",
                "max_tokens": 16,
                "temperature": 0.2,
                "stream": False,
            },
        )
    )

    for name, path, payload in tests:
        do_post(name, path, payload)


if __name__ == "__main__":
    main()
