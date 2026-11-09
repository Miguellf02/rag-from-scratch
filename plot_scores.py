"""Draws, for a question, the similarity score of every chunk and highlights the TOP_K that the RAG picks.

Saves one SVG per question in assets/ (plain SVG written by hand, no plotting library).
Run it from the project root with: uv run python plot_scores.py
"""
import math
from html import escape
from pathlib import Path

import numpy as np

import rag
from tests.cases import topic_of

QUESTIONS = {
    "scores_bb84.svg": "What is the QBER threshold for BB84?",
    "scores_france.svg": "What is the capital of France?",
}
OUT = Path(__file__).parent / "assets"

# Layout, in pixels
WIDTH, LABEL_W, PLOT_X, PLOT_W = 720, 230, 240, 400
TOP, ROW_H, BAR_H = 100, 28, 18

# Colors live in CSS variables, so the chart follows the reader's light or dark mode.
STYLE = """
svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
      --axis:#c3c2b7; --picked:#2a78d6; --other:#898781;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
@media (prefers-color-scheme: dark) {
  svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835; --picked:#3987e5; }
}
.bg { fill: var(--surface); }
.title { fill: var(--ink); font-size: 16px; font-weight: 600; }
.sub { fill: var(--ink2); font-size: 13px; }
.topic { fill: var(--ink2); font-size: 13px; }
.topic.picked, .value.picked { fill: var(--ink); font-weight: 600; }
.value { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.bar { fill: var(--other); }
.bar.picked { fill: var(--picked); }
"""


def bar_path(x: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Function that draws a bar with a rounded tip and a square end at the baseline (as an SVG path)."""
    r = min(r, w, h / 2)
    return f"M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z"


def chart(question: str, chunks: list[str], vectors: np.ndarray) -> tuple[str, list[tuple[str, float]]]:
    """Function that scores every chunk against the question and returns the SVG plus the TOP_K picked
    (as (topic, score) pairs). It reuses rag.retrieve with k = all chunks, so the ranking is exactly the RAG's.
    """
    ranked = rag.retrieve(question, chunks, vectors, len(chunks))
    # Same 0-1 axis on every chart (1 = identical meaning), so charts can be compared side by side.
    x_max = max(1.0, math.ceil(max(s for _, s in ranked) * 10) / 10)
    to_x = lambda s: PLOT_X + max(s, 0) / x_max * PLOT_W  # noqa: E731  score -> pixel
    bottom = TOP + len(ranked) * ROW_H
    height = bottom + 40

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'role="img" aria-label="Similarity of every chunk to: {escape(question)}">',
        f"<style>{STYLE}</style>",
        f'<rect class="bg" width="{WIDTH}" height="{height}" rx="8"/>',
        f'<text class="title" x="16" y="30">{escape(question)}</text>',
        f'<text class="sub" x="16" y="52">Cosine similarity of every chunk to the question. '
        f"The top {rag.TOP_K} are sent to the model.</text>",
        # Legend: two swatches, so the meaning never relies on color alone
        f'<rect class="bar picked" x="16" y="68" width="12" height="12" rx="2"/>'
        f'<text class="sub" x="34" y="79">Sent to the model (top {rag.TOP_K})</text>',
        '<rect class="bar" x="230" y="68" width="12" height="12" rx="2"/>'
        '<text class="sub" x="248" y="79">Not retrieved</text>',
    ]
    for t in np.arange(0, x_max + 1e-9, 0.2):  # vertical gridlines + tick labels
        x = to_x(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" y2="{bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">{t:.1f}</text>')
    parts.append(f'<line class="axis" x1="{PLOT_X}" y1="{TOP - 6}" x2="{PLOT_X}" y2="{bottom}"/>')

    for row, (i, score) in enumerate(ranked):
        cls = " picked" if row < rag.TOP_K else ""
        y = TOP + row * ROW_H
        mid = y + ROW_H / 2
        parts.append(f'<text class="topic{cls}" x="{PLOT_X - 10}" y="{mid + 4:.1f}" text-anchor="end">{escape(topic_of(chunks[i]))}</text>')
        parts.append(f'<path class="bar{cls}" d="{bar_path(PLOT_X, mid - BAR_H / 2, to_x(score) - PLOT_X, BAR_H)}"/>')
        parts.append(f'<text class="value{cls}" x="{to_x(score) + 6:.1f}" y="{mid + 4:.1f}">{score:.3f}</text>')
    parts.append("</svg>")

    picked = [(topic_of(chunks[i]), s) for i, s in ranked[: rag.TOP_K]]
    return "\n".join(parts), picked


def main() -> None:
    """Function that embeds the docs once, draws one chart per question and prints which chunks were picked."""
    chunks = rag.load_chunks(Path(__file__).parent / "docs")
    vectors = rag.embed(chunks)
    OUT.mkdir(exist_ok=True)
    for filename, question in QUESTIONS.items():
        svg, picked = chart(question, chunks, vectors)
        (OUT / filename).write_text(svg, encoding="utf-8")
        print(f"{question}\n  " + "\n  ".join(f"{t} ({s:.3f})" for t, s in picked) + f"\n  -> assets/{filename}")


if __name__ == "__main__":
    main()
