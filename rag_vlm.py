"""
VLM RAG -- ask a question, it RETRIEVES the relevant figure via CLIP, then answers
by reasoning over the actual image (gpt-4o vision) plus the retrieved text context.

This is the counterpart to rag_multimodal: that system passes the figure's CAPTION
to a text model, whereas this one passes the figure's PIXELS to a vision model. The
two are compared head-to-head in compare_multimodal.py, so the accuracy gap isolates
what the caption loses versus reading the image directly.

Built-in EXPLAINABILITY (shared explainability.py wrapper), at a level set with --level N:
    python rag_vlm.py --level 1   # one-line provenance
    python rag_vlm.py --level 2   # sources + retrieval + reasoning (default)
    python rag_vlm.py --level 3   # full trail incl. snippets/caption
    python rag_vlm.py --level 0   # answer only

Setup: gpt-4o vision via OPENAI_API_KEY; existing CLIP/index deps.
"""
import os
import io
import base64

from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

from rag_basics import build_index, retrieve
from rag_multimodal import build_image_index, retrieve_images
# explainability wrapper, shared by all systems
from explainability import format_explanation, text_evidence, parse_level

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
VLM_MODEL = "gpt-4o"
IMAGE_DIR = "publaynet_images"
K_TEXT = 3
K_IMG = 1


def b64_image(path):
    # encode a PNG as base64 so it can go inline in the vision request
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def answer_vlm(query, text_retrieved, image_name):
    """VLM answer from the retrieved image plus the retrieved text context."""
    text_ctx = "\n\n".join(f"[{source}] {chunk[:400]}" for _, source, chunk in text_retrieved)
    b64 = b64_image(os.path.join(IMAGE_DIR, image_name))
    resp = client.chat.completions.create(
        model=VLM_MODEL, temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text":
                 "Answer the QUESTION using the provided IMAGE (a figure/table from a "
                 "scientific paper) together with the TEXT CONTEXT. Read values, trends, "
                 "rows/columns, or comparisons directly from the image where relevant. "
                 "If the answer is in neither, say you don't know.\n\n"
                 f"TEXT CONTEXT:\n{text_ctx}\n\n"
                 f"QUESTION: {query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    return resp.choices[0].message.content


def ask_vlm(query, text_index, chunks, sources, img_index, img_names, captions):
    text_retrieved = retrieve(query, text_index, chunks, sources, k=K_TEXT)
    images = retrieve_images(query, img_index, img_names, captions, k=K_IMG)
    if not images:
        return "No figure retrieved.", text_retrieved, None, None, None
    score, image_name, caption = images[0]
    answer = answer_vlm(query, text_retrieved, image_name)
    return answer, text_retrieved, image_name, score, caption


if __name__ == "__main__":
    LEVEL = parse_level(default=2)          # read --level N once
    text_index, chunks, sources = build_index()
    img_index, img_names, captions = build_image_index()
    print(f"\nVLM RAG (explain level={LEVEL}) -- ask a question; it finds the figure and reads it.")
    print("Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        answer, text_retrieved, image_name, score, caption = ask_vlm(
            question, text_index, chunks, sources, img_index, img_names, captions
        )

        # evidence dict assembled from what this system actually retrieved
        evidence = {
            "text": text_evidence(text_retrieved),        # retrieved text chunks
            "figure": ({"image": image_name, "caption": caption or "", "score": score}
                       if image_name else None),          # retrieved figure
            "reasoning": "Answer read from the retrieved figure together with the text context.",
        }
        print("\n" + format_explanation(answer, evidence, LEVEL) + "\n" + "-" * 64)