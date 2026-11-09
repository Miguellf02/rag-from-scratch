"""Asks the RAG every question in cases.py and writes what it retrieved and answered to tests/results.md.

Run it from the project root with: uv run python -m tests.run_examples
"""
from datetime import date
from pathlib import Path

import rag
from tests.cases import CASES, topic_of

OUT = Path(__file__).parent / "results.md"


def main() -> None:
    """Function that asks every question to the real RAG and saves what it found and answered in results.md."""
    chunks = rag.load_chunks(Path(__file__).parent.parent / "docs")
    vectors = rag.embed(chunks)
    lines = [
        "# Example results",
        "",
        f"Generated on {date.today()} with `{rag.EMBED_MODEL}` and `{rag.CHAT_MODEL}`, top {rag.TOP_K} chunks.",
        "The chat model is not deterministic, so answers can change slightly between runs.",
        "Regenerate with `uv run python -m tests.run_examples`.",
        "",
    ]
    for case in CASES:
        hits = rag.retrieve(case["question"], chunks, vectors, rag.TOP_K)
        answer = rag.ask_llm(rag.build_prompt(case["question"], [chunks[i] for i, _ in hits]))
        expected = case["expected_topic"] or "nothing relevant"
        lines += [
            f"## {case['question']}",
            "",
            f"- **Expected:** {expected}",
            "- **Retrieved:** " + ", ".join(f"{topic_of(chunks[i])} ({s:.3f})" for i, s in hits),
            "",
            "**Answer:**",
            "",
            *[f"> {line}" if line else ">" for line in answer.strip().splitlines()],
            "",
        ]
        print(f"done: {case['question']}")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
