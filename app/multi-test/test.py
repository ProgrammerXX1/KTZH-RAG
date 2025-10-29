#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, asyncio, argparse, time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import httpx

DEFAULT_URL = "http://92.46.59.74:3001/rag/answer"
SCRIPT_DIR = Path(__file__).parent

# === Строгий payload: меняем ТОЛЬКО допустимые поля ===
def make_payload(
    query: str,
    doc_id: Optional[str],
    language: str,
    k_rules: int,
    k_defs: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "language": language,
        "k_rules": int(k_rules),
        "k_defs": int(k_defs),
        "need_versions_note": True,
    }
    if doc_id and doc_id.lower() not in {"string", "none", "auto", ""}:
        payload["doc_id"] = doc_id
    return payload

# --- Извлечение ответа и чанков из типичных схем ---
def extract_answer_and_chunks(resp_json: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    answer: Optional[str] = None

    # Популярные поля
    for k in ("answer", "result", "output", "text"):
        v = resp_json.get(k)
        if isinstance(v, str):
            answer = v
            break

    # OpenAI-like
    if answer is None and isinstance(resp_json.get("choices"), list) and resp_json["choices"]:
        ch0 = resp_json["choices"][0]
        if isinstance(ch0, dict):
            if isinstance(ch0.get("message"), dict) and isinstance(ch0["message"].get("content"), str):
                answer = ch0["message"]["content"]
            elif isinstance(ch0.get("text"), str):
                answer = ch0["text"]

    # Иногда бывает data.answer
    if answer is None and isinstance(resp_json.get("data"), dict):
        v = resp_json["data"].get("answer")
        if isinstance(v, str):
            answer = v

    # evidence → компактные элементы для анализа
    chunks: List[Dict[str, Any]] = []
    ev = resp_json.get("evidence")
    if isinstance(ev, list):
        for e in ev:
            if not isinstance(e, dict):
                continue
            text = e.get("text") or e.get("chunk_text")
            if not text and isinstance(e.get("chunk"), dict):
                text = e["chunk"].get("text") or e["chunk"].get("content")
            cite = e.get("cite") or e.get("id") or e.get("ref")

            # если cite нет — собрать из doc/loc
            if not cite and isinstance(e.get("doc"), dict) and isinstance(e.get("loc"), dict):
                d_id = e["doc"].get("id") or e["doc"].get("title")
                loc = []
                for kk in ("section_number", "chunk_rule_number", "page_number", "page", "chunk_index"):
                    if kk in e["loc"] and e["loc"][kk] not in (None, ""):
                        loc.append(f"{kk}={e['loc'][kk]}")
                if d_id:
                    cite = f"{d_id}|{','.join(loc)}" if loc else str(d_id)

            item: Dict[str, Any] = {}
            if text:
                item["text"] = text
            if cite:
                item["cite"] = cite
            if item:
                chunks.append(item)

    # Альтернативное поле "chunks"
    if not chunks and isinstance(resp_json.get("chunks"), list):
        for t in resp_json["chunks"]:
            if isinstance(t, str):
                chunks.append({"text": t})
            elif isinstance(t, dict):
                it = {}
                if isinstance(t.get("text"), str):
                    it["text"] = t["text"]
                if isinstance(t.get("cite"), str):
                    it["cite"] = t["cite"]
                if it:
                    chunks.append(it)

    return answer, chunks

async def ask_one(
    client: httpx.AsyncClient,
    url: str,
    q: str,
    headers: Dict[str, str],
    timeout: float,
    retries: int,
    idx: int,
    total: int,
    doc_id: Optional[str],
    language: str,
    k_rules: int,
    k_defs: int,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            payload = make_payload(q, doc_id, language, k_rules, k_defs)
            # лёгкий лог только в stderr
            print(f"[{idx}/{total}] → POST {url}  :: {q[:70]}...", file=sys.stderr)
            r = await client.post(url, json=payload, headers=headers, timeout=timeout)
            print(f"[{idx}/{total}] ← status={r.status_code}", file=sys.stderr)

            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except Exception as e:
                    print(f"[{idx}/{total}] JSON parse error: {repr(e)}", file=sys.stderr)
                    data = {"text": r.text}

                answer, chunks = extract_answer_and_chunks(data)
                return {"question": q, "answer": answer, "chunks": chunks}

            else:
                snippet = (r.text or "")[:300].replace("\n", " ")
                print(f"[{idx}/{total}] HTTP {r.status_code}: {snippet}", file=sys.stderr)
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
            print(f"[{idx}/{total}] ❌ Exception: {last_err}", file=sys.stderr)
        await asyncio.sleep(0.6 * (attempt + 1))

    return {"question": q, "answer": None, "chunks": []}

async def run_batch(
    url: str,
    questions: List[str],
    api_key: Optional[str],
    concurrency: int,
    timeout: float,
    retries: int,
    out_jsonl: str,
    doc_id: Optional[str],
    language: str,
    k_rules: int,
    k_defs: int,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    start_time = time.time()
    async with httpx.AsyncClient(timeout=None) as client:
        async def worker(q: str, idx: int):
            async with sem:
                return await ask_one(
                    client, url, q, headers, timeout, retries, idx, len(questions),
                    doc_id, language, k_rules, k_defs
                )

        tasks = [asyncio.create_task(worker(q, i + 1)) for i, q in enumerate(questions)]
        for t in asyncio.as_completed(tasks):
            res = await t
            results.append(res)

    elapsed = time.time() - start_time

    # Пишем только question/answer/chunks
    out_dir = os.path.dirname(out_jsonl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {"question": r.get("question"), "answer": r.get("answer"), "chunks": r.get("chunks", [])},
                    ensure_ascii=False,
                )
                + "\n"
            )

    ok = sum(1 for r in results if r.get("answer"))
    print("\n" + "=" * 80)
    print(f"Готово: {ok}/{len(results)} ответов за {elapsed:.2f} сек.")
    print(f"JSONL: {out_jsonl}")
    print("=" * 80 + "\n")

def load_questions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        return list(map(str, data["questions"]))
    if isinstance(data, list):
        return list(map(str, data))
    raise ValueError("Unsupported questions file format")

def main():
    p = argparse.ArgumentParser(description="RAG batch: only question/answer/chunks")
    p.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL (POST)")
    p.add_argument("--questions", default=str((SCRIPT_DIR / "questions.json").resolve()), help="Path to JSON file with questions")
    p.add_argument("--api-key", default=os.getenv("RAG_API_KEY"), help="API key (Authorization/X-API-Key)")
    p.add_argument("--doc-id", default=os.getenv("RAG_DOC_ID", ""), help="payload.doc_id. Пусто — без фильтра по документу")
    p.add_argument("--language", default=os.getenv("RAG_LANG", "ru"), help="payload.language")
    p.add_argument("--k-rules", type=int, default=int(os.getenv("RAG_K_RULES", "100")), help="payload.k_rules")
    p.add_argument("--k-defs", type=int, default=int(os.getenv("RAG_K_DEFS", "60")), help="payload.k_defs")
    p.add_argument("--concurrency", type=int, default=6, help="Max concurrent requests")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds)")
    p.add_argument("--retries", type=int, default=3, help="Retries per question")
    p.add_argument("--out-jsonl", default=str((SCRIPT_DIR / "rag_results.jsonl").resolve()), help="Output JSONL path")

    args = p.parse_args()

    print("Запуск батч-теста RAG")
    print(f"URL: {args.url}")
    print(f"Файл вопросов: {args.questions}")
    print(f"doc_id: {args.doc_id or '(none)'}")
    print(f"language: {args.language}  k_rules: {args.k_rules}  k_defs: {args.k_defs}\n")

    qs = load_questions(args.questions)
    print(f"Всего вопросов: {len(qs)}\n")

    asyncio.run(
        run_batch(
            args.url, qs, args.api_key, args.concurrency, args.timeout, args.retries,
            args.out_jsonl, args.doc_id, args.language, args.k_rules, args.k_defs
        )
    )

if __name__ == "__main__":
    main()
