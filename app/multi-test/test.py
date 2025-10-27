#!/usr/bin/env python3
import os, sys, json, asyncio, argparse, time
from typing import List, Dict, Any, Optional
from pathlib import Path
import httpx

DEFAULT_URL = "http://92.46.59.74:3001/rag/answer"
SCRIPT_DIR = Path(__file__).parent

# === Строгий payload: меняем ТОЛЬКО "query" ===
def make_payload(query: str, doc_id: str) -> Dict[str, Any]:
    return {
        "query": query,
        "language": "ru",
        "k_rules": 100,
        "k_defs": 60,
        "need_versions_note": True,
        "doc_id": doc_id,   # по умолчанию можно оставить "string" или пробросить через --doc-id
    }

# --- Извлечение ответа и чанков из типичных схем ---
def extract_answer_and_chunks(resp_json: Dict[str, Any]) -> (Optional[str], List[Dict[str, Any]]):
    answer: Optional[str] = None

    # Популярные поля
    for k in ("answer", "result", "output", "text"):
        if isinstance(resp_json.get(k), str):
            answer = resp_json.get(k)
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

    # Вытаскиваем минимальные "chunks"/evidence
    chunks: List[Dict[str, Any]] = []

    ev = resp_json.get("evidence")
    if isinstance(ev, list):
        for e in ev:
            if not isinstance(e, dict):
                continue
            text = None
            cite = None
            if isinstance(e.get("chunk"), dict):
                text = e["chunk"].get("text") or e["chunk"].get("content")
            text = text or e.get("text") or e.get("chunk_text")
            cite = e.get("cite") or e.get("id") or e.get("ref")

            # doc+loc → компактная ссылка
            if not cite and isinstance(e.get("doc"), dict) and isinstance(e.get("loc"), dict):
                doc_id = e["doc"].get("id") or e["doc"].get("title")
                loc = []
                for kk in ("section_number","page","chunk_rule_number","line_from","line_to","chunk_index"):
                    if kk in e["loc"]:
                        loc.append(f"{kk}={e['loc'][kk]}")
                if doc_id:
                    cite = f"{doc_id}|{','.join(loc)}" if loc else str(doc_id)

            item: Dict[str, Any] = {}
            if text: item["text"] = text
            if cite: item["cite"] = cite
            if item: chunks.append(item)

    # Альтернативное поле "chunks"
    if not chunks and isinstance(resp_json.get("chunks"), list):
        for t in resp_json["chunks"]:
            if isinstance(t, str):
                chunks.append({"text": t})
            elif isinstance(t, dict):
                it = {}
                if isinstance(t.get("text"), str): it["text"] = t["text"]
                if isinstance(t.get("cite"), str): it["cite"] = t["cite"]
                if it: chunks.append(it)

    return answer, chunks

async def ask_one(
    client: httpx.AsyncClient, url: str, q: str, headers: Dict[str, str],
    timeout: float, retries: int, idx: int, total: int, doc_id: str
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            payload = make_payload(q, doc_id)
            # лёгкий лог только в stderr (в файл ничего лишнего не попадёт)
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
                # Возвращаем только нужные поля — всё остальное не попадёт в JSONL
                return {"question": q, "answer": answer, "chunks": chunks}

            else:
                # покажем кусок тела ошибки в stderr
                snippet = (r.text or "")[:300].replace("\n", " ")
                print(f"[{idx}/{total}] HTTP {r.status_code}: {snippet}", file=sys.stderr)
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = repr(e)
            print(f"[{idx}/{total}] ❌ Exception: {last_err}", file=sys.stderr)
        await asyncio.sleep(0.3 * (attempt + 1))

    # даже при ошибке возвращаем минимальную форму
    return {"question": q, "answer": None, "chunks": []}

async def run_batch(
    url: str, questions: List[str], api_key: Optional[str],
    concurrency: int, timeout: float, retries: int,
    out_jsonl: str, doc_id: str
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    start_time = time.time()
    async with httpx.AsyncClient() as client:
        async def worker(q: str, idx: int):
            async with sem:
                return await ask_one(client, url, q, headers, timeout, retries, idx, len(questions), doc_id)

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
            f.write(json.dumps({
                "question": r.get("question"),
                "answer": r.get("answer"),
                "chunks": r.get("chunks", [])
            }, ensure_ascii=False) + "\n")

    ok = sum(1 for r in results if r.get("answer"))
    print("\n" + "=" * 80)
    print(f"📊 Готово: {ok}/{len(results)} ответов за {elapsed:.2f} сек.")
    print(f"📁 JSONL: {out_jsonl}")
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
    p.add_argument("--questions", default=str((SCRIPT_DIR / "questions.json").resolve()),
                   help="Path to JSON file with questions")
    p.add_argument("--api-key", default=os.getenv("RAG_API_KEY"),
                   help="API key (Authorization/X-API-Key)")
    p.add_argument("--doc-id", default=os.getenv("RAG_DOC_ID", "string"),
                   help="Value for payload.doc_id")
    p.add_argument("--concurrency", type=int, default=5, help="Max concurrent requests")
    p.add_argument("--timeout", type=float, default=35.0, help="Per-request timeout (seconds)")
    p.add_argument("--retries", type=int, default=2, help="Retries per question")
    p.add_argument("--out-jsonl", default=str((SCRIPT_DIR / "rag_results.jsonl").resolve()),
                   help="Output JSONL path")

    args = p.parse_args()

    print("🚀 Запуск батч-теста RAG")
    print(f"URL: {args.url}")
    print(f"Файл вопросов: {args.questions}")
    print(f"doc_id: {args.doc_id}\n")

    qs = load_questions(args.questions)
    print(f"Всего вопросов: {len(qs)}\n")

    asyncio.run(run_batch(
        args.url, qs, args.api_key, args.concurrency, args.timeout, args.retries,
        args.out_jsonl, args.doc_id
    ))

if __name__ == "__main__":
    main()
