"""
Generate figure/table questions for the evaluation.

Each question is written as a plain scientific question whose answer happens to
live in a figure or table, WITHOUT revealing that the answer is visual (no "table",
"figure", "shown", "in the study", etc.). This is deliberate: it forces the systems
to work out for themselves that visual content is needed, which makes the
baseline-vs-multimodal comparison a fair test rather than one cued by the wording.

For each figure the model is given its caption (so it knows the content) and asked
for one such question plus a short answer. Output is tagged "type": "figure" with
the image filename as the source.

Usage:
    python generate_figure_questions.py
    # then skim the output: keep questions that read naturally AND are still
    # specific enough to point at the right figure; hand-edit any that slipped.
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o-mini"
IMAGE_DIR = "publaynet_images"
OUT_FILE = "questions_publaynet_figures.json"

# Few-shot block: shows the model the target style (direct scientific questions)
# and the framing words to avoid. The "bad -> good" pairs are the cases that kept
# slipping through (e.g. "...in the table", "...described by participants").
FEWSHOT = """Write the question as a DIRECT scientific question, exactly the way one
would ask about a fact stated in prose. The answer happens to live in a figure/table,
but the QUESTION must not reveal that in any way.

FORBIDDEN: do not mention "table", "figure", "diagram", "chart", "graph", "panel",
"image", "shown", "presented", and do NOT use framing phrases like "in the study",
"described by participants", "compared in", "listed", "according to the data". Ask the
bare scientific question only.

These are the STYLE TARGET (note: direct factual questions, no framing):
- {"q": "What is of paramount importance for the activation of p70S6K?", "answer": "the phosphorylation of Serine residue in 411 position"}
- {"q": "How many genes are included in the Retinitis Pigmentosa panel?", "answer": "132 genes"}
- {"q": "What are the categories of UTI symptoms?", "answer": "..."}
- {"q": "What types of images are used for the go and no-go trials?", "answer": "grapes and socks for go trials; cookies and socks for no-go trials"}
- {"q": "What lipid-lowering drugs act on cholesterol metabolism?", "answer": "statins, fibrates, and PCSK9 inhibitors"}

Bad (rejected) phrasings and their fixes:
- BAD: "What categories are used to classify the synthetic microbial communities in the table?"
  GOOD: "What categories are used to classify the synthetic microbial communities?"
- BAD: "What are the categories of UTI symptoms described by participants?"
  GOOD: "What are the categories of UTI symptoms?"
- BAD: "What conditions are compared for the percentage of labeled cells in the study?"
  GOOD: "What conditions affect the percentage of labeled cells?"
"""


def generate_qa(caption):
    """Ask the model for one modality-free question + answer from a figure caption.
    Returns {"q", "answer"} or None if the caption is too vague (model replies SKIP)."""
    prompt = (
        FEWSHOT +
        "\nNow read the CAPTION of a scientific figure/table below and write ONE "
        "natural, modality-free question whose answer is contained in that figure, "
        "plus its short answer.\n"
        "Rules:\n"
        "- The question must NOT mention table/figure/diagram/chart/panel/image/shown/presented.\n"
        "- The question must NOT use framing phrases like 'in the study', 'by participants', 'compared', 'listed', 'according to the data'.\n"
        "- Ask the bare scientific question, exactly as one would about a fact in prose.\n"
        "- The answer must be supported by the figure's content (the caption).\n"
        "- Make the question specific enough in TOPIC to identify this figure, "
        "without naming the modality.\n"
        "- If the caption is too vague to support a specific question, respond "
        '{"q":"SKIP","answer":"SKIP"}.\n'
        '- Respond with ONLY JSON: {"q":"...","answer":"..."}\n\n'
        f"CAPTION:\n{caption}"
    )
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.4,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if result.get("q", "SKIP") == "SKIP":
            return None
        return {"q": result["q"], "answer": result["answer"]}
    except Exception:
        return None


def main():
    with open(os.path.join(IMAGE_DIR, "captions.json"), encoding="utf-8") as f:
        captions = json.load(f)

    questions = []
    for image_name, caption in captions.items():
        qa = generate_qa(caption)
        if qa is None:
            print(f"[skip] {image_name}: caption too vague")
            continue
        questions.append({
            "q": qa["q"], "source": image_name, "answer": qa["answer"], "type": "figure",
        })
        print(f"[ok] {image_name}\n     Q: {qa['q']}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(questions)} figure questions to {OUT_FILE}")
    print("Skim: keep questions that read naturally and stay specific enough to point "
          "at the right figure; hand-edit any that became too vague or slipped in a "
          "modality word.")


if __name__ == "__main__":
    main()