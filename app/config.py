import os
from dotenv import load_dotenv

load_dotenv()

# === Weaviate ===
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate-test:8000")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# === Эмбеддер ===
# если хочешь работать через API (как у тебя на 92.46.59.74:8080)
EMBEDDER_API_BASE = os.getenv("EMBEDDER_API_BASE", "http://92.46.59.74:8000/v1")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY", "local")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "e5-multilingual-large")

# === LLM ===
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://92.46.59.74:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local")
LLM_MODEL = os.getenv("LLM_MODEL", "gptoss-120b")

# === Прочее ===
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "ru")
CTX_BUDGET_TOKENS = int(os.getenv("CTX_BUDGET_TOKENS", "14000"))

# main-full
HYBRID_ALPHA_SHORT = float(os.getenv("HYBRID_ALPHA_SHORT", "0.86"))
HYBRID_ALPHA_LONG  = float(os.getenv("HYBRID_ALPHA_LONG",  "0.54"))
DOC_TOPN           = int(os.getenv("DOC_TOPN", "2"))     # было 12
MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.58"))
MMR_K              = int(os.getenv("MMR_K", "96"))       # было 96
ONLY_LATEST_REV    = os.getenv("ONLY_LATEST_REV", "1") == "1"
EVIDENCE_HEADROOM  = int(os.getenv("EVIDENCE_HEADROOM", "600"))
MONODOC_MARGIN     = float(os.getenv("MONODOC_MARGIN", "0.35"))  # было 0.08

# --- Rewriter ---
REWRITER_ON         = os.getenv("REWRITER_ON", "1") == "1"
REWRITER_TEMP       = float(os.getenv("REWRITER_TEMP", "0.1"))
REWRITER_MAX_TOKENS = int(os.getenv("REWRITER_MAX_TOKENS", "256"))
REWRITER_WEIGHT     = float(os.getenv("REWRITER_WEIGHT", "0.9"))  # вклад переписанного поиска в скор

# medium-hign config
# # --- Retrieval knobs ---
# HYBRID_ALPHA_SHORT = float(os.getenv("HYBRID_ALPHA_SHORT", "0.82"))  # короткие запросы <5 слов
# HYBRID_ALPHA_LONG  = float(os.getenv("HYBRID_ALPHA_LONG",  "0.58"))  # длинные запросы
# DOC_TOPN           = int(os.getenv("DOC_TOPN", "12"))                 # сколько doc_id оставлять
# MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.6"))          # 1.0=макс.разнообразие, 0.0=макс.сходство
# MMR_K              = int(os.getenv("MMR_K", "96"))                   # сколько кандидатов после MMR
# ONLY_LATEST_REV    = os.getenv("ONLY_LATEST_REV", "1") == "1"        # отдавать только последнюю редакцию на today
# EVIDENCE_HEADROOM  = int(os.getenv("EVIDENCE_HEADROOM", "800"))      # запас токенов под сам ответ
# MONODOC_MARGIN = float(os.getenv("MONODOC_MARGIN", "0.08"))  # 15% запас по score для уверенного монодока


# medium config
# --- Retrieval knobs ---
# HYBRID_ALPHA_SHORT = float(os.getenv("HYBRID_ALPHA_SHORT", "0.82"))  # короткие запросы <5 слов
# HYBRID_ALPHA_LONG  = float(os.getenv("HYBRID_ALPHA_LONG",  "0.58"))  # длинные запросы
# DOC_TOPN           = int(os.getenv("DOC_TOPN", "8"))                 # сколько doc_id оставлять
# MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.65"))          # 1.0=макс.разнообразие, 0.0=макс.сходство
# MMR_K              = int(os.getenv("MMR_K", "32"))                   # сколько кандидатов после MMR
# ONLY_LATEST_REV    = os.getenv("ONLY_LATEST_REV", "1") == "1"        # отдавать только последнюю редакцию на today
# EVIDENCE_HEADROOM  = int(os.getenv("EVIDENCE_HEADROOM", "800"))      # запас токенов под сам ответ
# MONODOC_MARGIN = float(os.getenv("MONODOC_MARGIN", "0.15"))  # 15% запас по score для уверенного монодока

# single config
# --- Retrieval knobs ---
# HYBRID_ALPHA_SHORT = float(os.getenv("HYBRID_ALPHA_SHORT", "0.82"))  # короткие запросы <5 слов
# HYBRID_ALPHA_LONG  = float(os.getenv("HYBRID_ALPHA_LONG",  "0.58"))  # длинные запросы
# DOC_TOPN           = int(os.getenv("DOC_TOPN", "2"))                 # сколько doc_id оставлять
# MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.65"))          # 1.0=макс.разнообразие, 0.0=макс.сходство
# MMR_K              = int(os.getenv("MMR_K", "16"))                   # сколько кандидатов после MMR
# ONLY_LATEST_REV    = os.getenv("ONLY_LATEST_REV", "1") == "1"        # отдавать только последнюю редакцию на today
# EVIDENCE_HEADROOM  = int(os.getenv("EVIDENCE_HEADROOM", "600"))      # запас токенов под сам ответ
# MONODOC_MARGIN = float(os.getenv("MONODOC_MARGIN", "0.25"))  # 15% запас по score для уверенного монодока
