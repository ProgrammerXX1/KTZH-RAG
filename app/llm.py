import json
import requests

BASE = "http://92.46.59.74:8000/v1"
API_KEY = "local"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def test_chat():
    print("=" * 60)
    print("TEST: /v1/chat/completions")

    payload = {
        "model": "gpt-oss-120b",
        "messages": [
            {"role": "user", "content": "ping from tester_chat.py"}
        ],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False
    }

    r = requests.post(f"{BASE}/chat/completions", headers=HEADERS, json=payload)

    print("Status:", r.status_code)
    print("Raw:", r.text[:500])

    try:
        print("JSON:", json.dumps(r.json(), ensure_ascii=False, indent=2))
    except:
        print("JSON PARSE ERROR")

def test_legacy_completions(session: requests.Session, base_url: str, headers: dict):
    name = "legacy_completions"
    url = f"{base_url}/v1/completions"
    # Жёсткий промпт, чтобы модель не фантазировала про endpoint
    payload = {
        "model": "gpt-oss-120b",
        "prompt": "Ответь строго одним словом: PONG. Ничего больше не пиши.",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": False,
    }
    print("=" * 80)
    print(f"TEST: {name}")
    print("URL: ", url)
    print("Payload:")
    print(" ", json.dumps(payload, ensure_ascii=False, indent=2))

    resp = session.post(url, headers=headers, json=payload)
    print("Status:", resp.status_code)
    text_raw = resp.text
    print("Raw response (first 800 chars):")
    print(" ", text_raw[:800])

    try:
        jj = resp.json()
        print("Parsed JSON:")
        print(" ", json.dumps(jj, ensure_ascii=False, indent=2))
        # Достаём сам ответ модели
        choice = (jj.get("choices") or [{}])[0]
        answer_text = (choice.get("text") or "").strip()
        print("Answer text:", repr(answer_text))
    except Exception:
        print("Parsed JSON: <cannot parse>")



if __name__ == "__main__":
    test_chat()
    test_completions()
