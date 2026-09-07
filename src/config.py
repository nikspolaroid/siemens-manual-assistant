"""All tunable settings in one place."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANUALS_DIR = ROOT / "data" / "manuals"
DB_PATH = ROOT / "chroma_db"

COLLECTION_NAME = "siemens_manuals"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 50

TOP_K = 6
EMBED_BATCH = 64
STORE_BATCH = 500
