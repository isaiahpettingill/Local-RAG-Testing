import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

KNOWLEDGEBASE_DB = DATA_DIR / "knowledgebase.db"
LADYBUGDB_DIR = DATA_DIR / "ladybugdb"
EVAL_QUEUE_DB = DATA_DIR / "eval_queue.db"
INGESTION_QUEUE_DB = DATA_DIR / "ingestion_queue.db"
STAGING_DB = DATA_DIR / "staging.db"
CRAWL_DB = DATA_DIR / "crawl.db"

LADYBUGDB_DIR.mkdir(exist_ok=True)

MODELS = {
    "embedding": {
        "name": "embeddinggemma-300M",
    },
    "agent": {
        "name": "Gemma 4 E2B 8bit GGML",
        "path": None,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_threads": None,
        "n_batch": 512,
    },
    "grader": {
        "name": "Gemma 4 26B",
        "path": None,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_threads": None,
        "n_batch": 512,
    },
    "graph_extractor": {
        "name": "Gemma 4 26B",
        "path": None,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_threads": None,
        "n_batch": 512,
    },
    "graph_summarizer": {
        "name": "Gemma 4 E4B-it",
        "path": None,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_threads": None,
        "n_batch": 256,
    },
}

SQLITE_VEC_DIMENSION = 384
RECIPROCAL_RANK_FUSION_LIMIT = 3

# Local LLM Routing
OPENAI_BASE_URL = "http://localhost:8080/v1"
OPENAI_API_KEY = "local-no-key"

os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL  # noqa: E402
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
