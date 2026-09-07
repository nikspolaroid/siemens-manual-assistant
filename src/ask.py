"""Retrieval-augmented Q&A over Siemens manuals."""
import sys

from config import TOP_K
from search import search
from llm import ask

SYSTEM = """You are a technical assistant for Siemens industrial automation
equipment. A maintenance technician is reading your answer at a machine.

Rules:
1. Answer ONLY from the manual excerpts provided. Never use outside knowledge.
2. Cite every factual claim as [filename, p. N] from the excerpt headers.
3. If the excerpts do not answer the question, say exactly:
   "I could not find this in the loaded manuals."
   Then name the manual or section that would likely contain it. Never guess.
4. If the work involves live equipment, state that the machine must be
   isolated before work begins.
5. Be concise and practical. Prefer short steps over prose."""


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"--- [{h['source']}, p. {h['page']}] ---\n{h['text']}" for h in hits
    )


def answer(question: str, k: int = TOP_K, show_sources: bool = False) -> str:
    hits = search(question, k=k)
    prompt = (
        f"Manual excerpts:\n\n{build_context(hits)}\n\n"
        f"Technician's question: {question}"
    )
    reply = ask(SYSTEM, prompt)

    if show_sources:
        refs = "\n".join(
            f"  [{i}] {h['source']} p.{h['page']} (distance {h['distance']:.3f})"
            for i, h in enumerate(hits, 1)
        )
        reply += f"\n\nRetrieved passages:\n{refs}"
    return reply


def repl() -> None:
    print("Siemens Manual Assistant")
    print("Type a question, or 'sources' to toggle retrieved passages. "
          "Ctrl-C to quit.\n")
    show = False
    while True:
        try:
            q = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not q:
            continue
        if q.lower() == "sources":
            show = not show
            print(f"Source display: {'on' if show else 'off'}\n")
            continue
        print("\n" + answer(q, show_sources=show) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("\n" + answer(" ".join(sys.argv[1:]), show_sources=True) + "\n")
    else:
        repl()
    # ChromaDB and torch race during interpreter teardown on macOS.
    # Everything above has already run; skip destructors.
    import sys, os
    sys.stdout.flush()
    os._exit(0)
