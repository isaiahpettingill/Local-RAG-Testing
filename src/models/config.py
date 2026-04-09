from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

KNOWLEDGEBASE_DB = DATA_DIR / "knowledgebase.db"
LADYBUGDB_DIR = DATA_DIR / "ladybugdb"
EVAL_QUEUE_DB = DATA_DIR / "eval_queue.db"
INGESTION_QUEUE_DB = DATA_DIR / "ingestion_queue.db"

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
}

SQLITE_VEC_DIMENSION = 384
RECIPROCAL_RANK_FUSION_LIMIT = 3
