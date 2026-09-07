"""All tunable settings in one place. Env vars override for experiments."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANUALS_DIR = ROOT / "data" / "manuals"
DB_PATH = ROOT / "chroma_db"
EVAL_FILE = ROOT / "eval" / "questions.json"

COLLECTION_NAME = "siemens_manuals"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
MIN_PAGE_CHARS = 50

TOP_K = int(os.getenv("TOP_K", 6))
PAGE_TOLERANCE = 1

EMBED_BATCH = 64
STORE_BATCH = 500
