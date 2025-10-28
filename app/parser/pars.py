# parser_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Iterable
import tempfile, os, json, requests, time

# === LLM конфиг (как в config.py, но локально, если не импортируешь config)
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://92.46.59.74:8080/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "local")
LLM_MODEL    = os.getenv("LLM_MODEL",    "qwen3-next-80b-a3b")

# === Модели локальные для роутера
@dataclass
class Meta:
    source_document_name: str
    source_document_id: str
    document_title: str
    document_version: str
    effective_date: str
    department: Optional[str] = None
    language: str = "ru"  # "ru" | "kz"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

# === Извлечение текста БЕЗ правок
def _read_docx(path: Path) -> List[str]:
    from docx import Document
    doc = Document(str(path))
    # берём абзацы «как есть»; никаких склеек/дефисов/регексов
    return [p.text.replace("\u00A0", " ").rstrip() for p in doc.paragraphs if p.text and p.text.strip()]

def _read_pdf(path: Path) -> List[str]:
    import pdfplumber
    lines: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # сохраняем построчно, как дал движок
            lines.extend([ln.rstrip() for ln in text.splitlines()])
    # убираем только полностью пустые строки
    return [ln for ln in lines if ln is not None and ln.strip() != ""]

def collect_text_verbatim(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".docx":
        lines = _read_docx(path)
    elif ext == ".pdf":
        lines = _read_pdf(path)
    else:
        raise ValueError(f"unsupported: {ext}")
    # объединяем только символом перевода строки; это не меняет содержание строк
    return "\n".join(lines)

def batch_text_verbatim(big_text: str, max_chars: int) -> List[str]:
    # режем по символам, но только по границе \n чтобы не ломать строки
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for line in big_text.split("\n"):
        add_len = len(line) + (1 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(line); cur_len += add_len
        else:
            out.append("\n".join(cur))
            cur = [line]; cur_len = len(line)
    if cur:
        out.append("\n".join(cur))
    return out

# === Вызов LLM
def call_llm(batch_text: str, meta: Meta, timeout=120, retries=3) -> List[Dict[str, Any]]:
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}

    # строгий промпт: контент кусков — строго подстроки входа
    system_prompt = (
        "Ты конвертер текста в JSON. Режь входной текст на логические куски БЕЗ изменений.\n"
        "Никаких перефразирований, исправлений и добавлений. content каждого объекта должен быть "
        "буквальной подстрокой входа (verbatim). Верни ТОЛЬКО JSON-массив.\n"
        "Схема объекта:\n"
        "{\n"
        "  'chunk_id': str,\n"
        "  'content': str,                # точная подстрока из входа\n"
        "  'metadata': {\n"
        "    'source_document_name': str,\n"
        "    'source_document_id': str,\n"
        "    'document_title': str,\n"
        "    'document_version': str,\n"
        "    'effective_date': str,\n"
        "    'department': str|null,\n"
        "    'language': 'ru'|'kz',\n"
        "    'section_number': str|null,\n"
        "    'section_title': str|null,\n"
        "    'parent_rule_number': str|null,\n"
        "    'chunk_rule_number': str|null,\n"
        "    'page_number': null,\n"
        "    'chunk_type': 'paragraph'|'parent_rule'|'section_title',\n"
        "    'cross_references': []\n"
        "  }\n"
        "}\n"
        "Правила:\n"
        "1) content копируй из входа без изменений.\n"
        "2) Формируй chunk_id как '<doc_id>-<секция>' или '<doc_id>-<номер>' по смыслу. Уникальность обязательна в пределах документа.\n"
        "3) Если поле недоступно — ставь null, но обязательные поля из шапки документа заполняй из метаданных ниже.\n"
        "4) Возвращай только JSON-массив без обрамляющего текста и без кода.\n"
    )

    user_prompt = (
        f"Шапка документа:\n"
        f"source_document_name: {meta.source_document_name}\n"
        f"source_document_id: {meta.source_document_id}\n"
        f"document_title: {meta.document_title}\n"
        f"document_version: {meta.document_version}\n"
        f"effective_date: {meta.effective_date}\n"
        f"department: {meta.department}\n"
        f"language: {meta.language}\n\n"
        f"Текст к разрезанию:\n{batch_text}"
    )

    payload = {
        "model": LLM_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_err = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            r.raise_for_status()
            body = r.json()["choices"][0]["message"]["content"].strip()
            if body.startswith("```"):
                body = body.strip("`")
                if body.startswith("json\n"):
                    body = body[5:]
            data = json.loads(body)
            if not isinstance(data, list):
                raise ValueError("LLM returned non-list")
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise HTTPException(502, detail=f"llm_failed: {repr(last_err)}")

# === Пост-валидация: content должен быть подстрокой входа + заполнение обязательных полей меты
def normalize_objects(arr: List[Dict[str, Any]], meta: Meta, batch_src: str) -> Iterable[Dict[str, Any]]:
    for obj in arr:
        if not isinstance(obj, dict): 
            continue
        cid = obj.get("chunk_id")
        content = obj.get("content")
        md = obj.get("metadata") or {}
        if not cid or not isinstance(content, str) or not content:
            continue
        # жесткая проверка «вербатим»
        if content not in batch_src:
            continue  # отбрасываем любые «неподстроки»

        # заполняем обязательные поля схемы, если LLM их не проставил
        md.setdefault("source_document_name", meta.source_document_name)
        md.setdefault("source_document_id", meta.source_document_id)
        md.setdefault("document_title", meta.document_title)
        md.setdefault("document_version", meta.document_version)
        md.setdefault("effective_date", meta.effective_date)
        md.setdefault("department", meta.department)
        md.setdefault("language", meta.language)

        # прочие поля — пусть будут хотя бы с нулями по твоей модели
        md.setdefault("section_number", None)
        md.setdefault("section_title", None)
        md.setdefault("parent_rule_number", None)
        md.setdefault("chunk_rule_number", None)
        md.setdefault("page_number", None)
        md.setdefault("chunk_type", "paragraph")
        md.setdefault("cross_references", [])

        yield {"chunk_id": str(cid), "content": content, "metadata": md}

# === FastAPI
router = APIRouter(prefix="/parser", tags=["parser"])

@router.get("/health")
def health():
    return {"ok": True}

# NDJSON-стрим
@router.post("/parse")
async def parser_router(
    file: UploadFile = File(...),
    source_document_id: str = Form(None),   # можно не передавать
    language: str = Form("ru"),             # для эмбеддера достаточно
    max_chars: int = Form(6000),
    timeout: int = Form(120),
    retries: int = Form(3),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx"}:
        raise HTTPException(400, detail="supported: .pdf, .docx")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        doc_id = source_document_id or Path(file.filename or tmp_path).stem
        # заполняем обязательные поля пустыми строками — эмбеддеру не мешает
        meta = Meta(
            source_document_name=file.filename or Path(tmp_path).name,
            source_document_id=doc_id,
            document_title="",
            document_version="",
            effective_date="",
            department=None,
            language=language,
        )

        full_text = collect_text_verbatim(Path(tmp_path))
        batches = batch_text_verbatim(full_text, max_chars=max_chars)

        def gen():
            for b in batches:
                arr = call_llm(b, meta, timeout=timeout, retries=retries)
                for row in normalize_objects(arr, meta, batch_src=b):
                    # row уже содержит {chunk_id, content, metadata{...}}
                    yield json.dumps(row, ensure_ascii=False) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")
    finally:
        try: os.unlink(tmp_path)
        except Exception: pass

# Выгрузка как файл .jsonl
@router.post("/parse/file")
async def parse_as_file(
    file: UploadFile = File(...),
    document_title: str = Form(...),
    document_version: str = Form(...),
    effective_date: str = Form(...),
    department: Optional[str] = Form(None),
    language: str = Form("ru"),
    source_document_id: Optional[str] = Form(None),
    max_chars: int = Form(6000),
    timeout: int = Form(120),
    retries: int = Form(3),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".docx"}:
        raise HTTPException(400, detail="supported: .pdf, .docx")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
        tmp_in.write(await file.read())
        in_path = tmp_in.name

    out_path = tempfile.mktemp(suffix=".jsonl")
    try:
        meta = Meta(
            source_document_name=file.filename or Path(in_path).name,
            source_document_id=source_document_id or Path(file.filename or in_path).stem,
            document_title=document_title,
            document_version=document_version,
            effective_date=effective_date,
            department=department,
            language=language,
        )
        full_text = collect_text_verbatim(Path(in_path))
        batches = batch_text_verbatim(full_text, max_chars=max_chars)

        with open(out_path, "w", encoding="utf-8") as f:
            for b in batches:
                arr = call_llm(b, meta, timeout=timeout, retries=retries)
                for row in normalize_objects(arr, meta, batch_src=b):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        data = Path(out_path).read_bytes()
        headers = {
            "Content-Disposition": f'attachment; filename="{meta.source_document_id}.jsonl"'
        }
        return Response(content=data, media_type="application/x-ndjson", headers=headers)
    finally:
        try: os.unlink(in_path)
        except Exception: pass
        try: os.unlink(out_path)
        except Exception: pass
