"""
Shared explainability wrapper, imported by every RAG system so they all explain
their answers the same way, inline, at a the choosen verbosity.

At answer time a system builds an `evidence` dict and calls
format_explanation(answer, evidence, level). The explanation prints as part of
that system's own output, so there is no separate explainability script to run.

Levels (set with --level N on any system; default 2):
  0  answer only (e.g. when running evaluation, where the explanation is noise)
  1  answer + a one-line note of the sources and how they were retrieved
  2  answer + sources, retrieval method, and a short reasoning line
  3  answer + the full trail, including text snippets and the figure caption

The evidence dict holds only what a given system actually used, e.g. the baseline
fills in "text" only, while +both fills in "text", "graph_facts", and "figure":
  {
    "text":        [{"source": ..., "chunk": ..., "score": ...}, ...],
    "graph_facts": ["subject relation object", ...],
    "figure":      {"image": ..., "caption": ..., "score": ...},
    "reasoning":   "one line tying the evidence to the answer"   (optional)
  }
"""
import sys


def parse_level(default=2):
    """Read --level N from the command line; fall back to the default if absent."""
    if "--level" in sys.argv:
        try:
            return int(sys.argv[sys.argv.index("--level") + 1])
        except (IndexError, ValueError):
            pass
    return default


def _provenance_line(evidence):
    """Build the compact one-line 'where from + how retrieved' string for level 1."""
    parts = []
    if evidence.get("text"):
        # unique source names, keeping first-seen order
        srcs = ", ".join(dict.fromkeys(t["source"] for t in evidence["text"]))
        parts.append(f"text[{srcs}]")
    if evidence.get("graph_facts"):
        parts.append(f"graph[{len(evidence['graph_facts'])} facts]")
    if evidence.get("figure"):
        fig = evidence["figure"]
        parts.append(f"figure[{fig['image']} @CLIP {fig.get('score', 0):.2f}]")
    return " + ".join(parts) if parts else "(no evidence)"


def format_explanation(answer, evidence, level=2):
    """Return the answer, plus an explanation rendered at the requested level."""
    if level <= 0:
        return answer

    lines = [answer, ""]

    if level == 1:
        lines.append("How I got this: " + _provenance_line(evidence))
        return "\n".join(lines)

    # level 2 and 3: itemised provenance
    lines.append("How I got this:")
    if evidence.get("text"):
        srcs = ", ".join(dict.fromkeys(t["source"] for t in evidence["text"]))
        lines.append(f"  - Text source(s): {srcs}  (retrieved by text similarity)")
    if evidence.get("graph_facts"):
        lines.append(f"  - Knowledge-graph facts used: {len(evidence['graph_facts'])}")
        if level >= 3:
            for fact in evidence["graph_facts"]:
                lines.append(f"      • {fact}")
    if evidence.get("figure"):
        fig = evidence["figure"]
        lines.append(f"  - Figure: {fig['image']}  (retrieved by CLIP, score {fig.get('score', 0):.3f})")
        if level >= 3:
            lines.append(f"      caption: {fig.get('caption', '')[:140]}")
    if evidence.get("reasoning"):
        lines.append(f"  - Reasoning: {evidence['reasoning']}")

    # level 3 only: show the actual retrieved text snippets
    if level >= 3 and evidence.get("text"):
        lines.append("  - Evidence snippets:")
        for t in evidence["text"]:
            lines.append(f"      [{t['source']}] {t['chunk'][:120]}...")

    return "\n".join(lines)


# ---------- small builders so each system fills the evidence dict consistently ----------
def text_evidence(retrieved):
    """Turn retrieve()'s (score, source, chunk) tuples into the evidence "text" list."""
    return [{"source": source, "chunk": chunk, "score": float(score)}
            for score, source, chunk in retrieved]

def figure_evidence(images):
    """Turn retrieve_images()'s (score, name, caption) list into figure evidence (top one)."""
    if not images:
        return None
    score, name, caption = images[0]
    return {"image": name, "caption": caption, "score": float(score)}