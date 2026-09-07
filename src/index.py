"""Embed all chunks locally and store them in ChromaDB."""
from config import DB_PATH, COLLECTION_NAME, EMBED_BATCH, STORE_BATCH
from db import get_client, get_collection, embed
from ingest import build_chunks


def main() -> None:
    chunks = build_chunks()
    if not chunks:
        raise SystemExit("No chunks to index. Check your PDFs first.")

    print(f"\nEmbedding {len(chunks)} chunks locally "
          f"(no document text leaves this machine)...")
    vectors = embed([c.text for c in chunks],
                    batch_size=EMBED_BATCH, progress=True)

    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared previous index.")
    except Exception:
        pass

    collection = get_collection()

    for i in range(0, len(chunks), STORE_BATCH):
        batch = chunks[i:i + STORE_BATCH]
        collection.add(
            ids=[f"c{i + j}" for j in range(len(batch))],
            embeddings=vectors[i:i + STORE_BATCH],
            documents=[c.text for c in batch],
            metadatas=[{"source": c.source, "page": c.page} for c in batch],
        )
        print(f"  stored {min(i + STORE_BATCH, len(chunks))}/{len(chunks)}")

    print(f"\nIndex built: {collection.count()} chunks at {DB_PATH}")


if __name__ == "__main__":
    main()
