"""LaTeX bodies of the manuscript's HotpotQA tables, from pooled-analysis reports.

    rag pooled-analysis <bridge traces> --questions data/questions/questions_hotpotqa_bridge.json \
        --split "+KGret=baseline" --split "+KG=dense-top3" --split "+HippoRAG=baseline" \
        --output artifacts/runs/pooled_hotpotqa_bridge.json
    (the same for comparison), then
    rag holm-family artifacts/runs/pooled_hotpotqa_bridge.json artifacts/runs/pooled_hotpotqa_comparison.json
    python docs/paper_tables.py artifacts/runs/pooled_hotpotqa_bridge.json \
        artifacts/runs/pooled_hotpotqa_comparison.json > tables.tex

Prints four blocks, marked %% BRIDGE, %% COMPARISON, %% DECOMP, %% PERGEN, each
the rows between a table's \\midrule and \\bottomrule.
"""

import json
import sys

ROWS = [
    ("dense-top3", "Dense top-3"),
    ("+KG", "+KG"),
    ("baseline", "Dense top-5"),
    ("bm25", "BM25 top-5"),
    ("hybrid-rrf", "Hybrid (RRF)"),
    ("+lexical", "Lexical expansion"),
    ("+random", "Random padding"),
    ("+HippoRAG", "HippoRAG\\,2"),
    ("+KGret", "+KGret"),
    None,
    ("control:oracle", "Oracle"),
    ("control:closed-book", "Closed-book"),
]
GENERATORS = [
    "llama4-scout",
    "gpt-6-luna",
    "gemma-4-31b",
    "gemini-3.8-flash",
    "qwen3.8-max",
]
SPLITS = [
    ("+KGret vs baseline", "+KGret minus dense top-5"),
    ("+KG vs dense-top3", "+KG minus dense top-3"),
    ("+HippoRAG vs baseline", "HippoRAG\\,2 minus dense top-5"),
]


def num(x, signed=False):
    if abs(x) < 0.0005:
        return "0.000"
    text = f"{x:+.3f}" if signed else f"{x:.3f}"
    return text.replace("-", "$-$").replace("+", "$+$")


def interval(d):
    return f"{{\\scriptsize[{num(d['lower'], True)}, {num(d['upper'], True)}]}}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main_table(report):
    lines = []
    for row in ROWS:
        if row is None:
            lines.append("\\midrule")
            continue
        key, label = row
        s = report["systems"][key]
        complete = "--" if key == "control:closed-book" else f"{s['complete']:.2f}"
        cells = [label, complete] + [num(s[m]) for m in ("em", "f1", "contains", "acc")]
        if key == report["reference"]:
            lines.append(" & ".join(cells + ["ref.", "--", "ref.", "--"]) + " \\\\")
            continue
        # the estimate on the row, its 95% interval on a small second line
        below = [""] * len(cells)
        for m in ("em", "acc"):
            d = s["differences"][m]
            cells += [
                num(d["delta"], True),
                pval(d["p_holm"]) if "p_holm" in d else "--",
            ]
            below += [interval(d), ""]
        lines.append(" & ".join(cells) + " \\\\")
        lines.append(" & ".join(below) + " \\\\[1pt]")
    return lines


def decomp_table(bridge, comparison):
    lines = []
    for b, (key, title) in enumerate(SPLITS):
        if b:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for report, label in ((bridge, "Bridge"), (comparison, "Comparison")):
            top, bottom = [label], [""]
            for m in ("em", "acc"):
                split = report["splits"][key][m]
                for term in ("delta", "access", "use"):
                    point, lo, hi = split[term]
                    if key.startswith("+KG vs") and term == "access":
                        top.append("0")
                        bottom.append("")
                    else:
                        top.append(num(point, True))
                        bottom.append(
                            f"{{\\scriptsize[{num(lo, True)}, {num(hi, True)}]}}"
                        )
            lines += [" & ".join(top) + " \\\\", " & ".join(bottom) + " \\\\"]
    return lines


def pergen_table(bridge, comparison):
    systems = [
        ("baseline", "Dense top-5"),
        ("+KGret", "+KGret"),
        ("+HippoRAG", "HippoRAG\\,2"),
        ("control:oracle", "Oracle"),
        ("control:closed-book", "Closed-book"),
    ]
    lines = []
    for b, (report, title) in enumerate(
        ((bridge, "Bridge"), (comparison, "Comparison"))
    ):
        if b:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for key, label in systems:
            per = report["per_generator"][key]
            cells = [label] + [
                f"{per[g]['em']:.2f} / {per[g]['contains']:.2f} / {per[g]['acc']:.2f}"
                for g in GENERATORS
            ]
            lines.append(" & ".join(cells) + " \\\\")
    return lines


if __name__ == "__main__":
    bridge = json.load(open(sys.argv[1], encoding="utf-8"))
    comparison = json.load(open(sys.argv[2], encoding="utf-8"))
    for name, lines in (
        ("BRIDGE", main_table(bridge)),
        ("COMPARISON", main_table(comparison)),
        ("DECOMP", decomp_table(bridge, comparison)),
        ("PERGEN", pergen_table(bridge, comparison)),
    ):
        print(f"%% {name}")
        print("\n".join(lines))
