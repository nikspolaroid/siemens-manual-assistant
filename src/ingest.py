"""Load Siemens PDFs and split them into page-tagged chunks."""
from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader

MANUALS_DIR = Path(__file__).parent.parent / "data" / "manuals"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def load_pdf(path: Path) -> list[tuple[str, int]]:
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("not a PDF (probably a saved HTML or error page)")
    reader = PdfReader(path)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            continue
        if len(text) > 50:
            pages.append((text, page_no))
    return pages


def split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return pieces


def build_chunks() -> list[Chunk]:
    chunks, skipped = [], []
    pdfs = sorted(MANUALS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {MANUALS_DIR}")
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        try:
            pages = load_pdf(pdf)
        except Exception as e:
            print(f"  SKIPPED - {e}")
            skipped.append(pdf.name)
            continue
        if not pages:
            print("  SKIPPED - no extractable text (scanned image?)")
            skipped.append(pdf.name)
            continue
        for page_text, page_no in pages:
            for piece in split(page_text):
                chunks.append(Chunk(piece, pdf.name, page_no))
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s): {', '.join(skipped)}")
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\nTotal chunks: {len(chunks)}\n")
    counts = {}
    for c in chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    for src, n in sorted(counts.items()):
        print(f"  {n:>6}  {src}")
    if chunks:
        print("\n--- sample chunk ---")
        print(f"[{chunks[0].source} p.{chunks[0].page}]")
        print(chunks[0].text[:400])
