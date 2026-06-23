"""
Full enhanced RAG = text + knowledge graph + multimodal, fused together.

This is the brief's "enhanced approach": it combines BOTH enhancements built and
measured separately, in a single generate call.

At query time it gathers three sources and injects all of them into the prompt:
  1. text chunks     (from rag_basics's retriever)
  2. graph facts     (from graph_aware's entity-neighbourhood lookup)
  3. image caption(s) (from rag_multimodal's CLIP image retrieval)

It reuses the components already built -- nothing here is new machinery, it's the
same fusion pattern applied to all three sources at once.

Setup: all previous deps (openai, faiss, networkx, torch, open-clip-torch, pillow).
"""
import os

from openai import OpenAI
from dotenv import load_dotenv

from rag_basics import build_index, retrieve, CHAT_MODEL
from graph_aware import build_triples, build_graph, graph_facts_for_query
from rag_multimodal import build_image_index, retrieve_images

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def generate_full(query, text_retrieved, graph_facts, image_retrieved):
    context = "\n\n".join(f"[{source}] {chunk}" for _, source, chunk in text_retrieved)
    facts_block = "\n".join(f"- {fact}" for fact in graph_facts) if graph_facts else "(none)"
    img_block = ("\n".join(f"- ({name}) {caption}" for _, name, caption in image_retrieved)
                 if image_retrieved else "(none)")

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "Answer using only the provided text context, knowledge-graph "
                        "facts, and image descriptions. If the answer isn't in any of "
                        "them, say you don't know."},
            {"role": "user",
             "content": (
                 f"Text context:\n{context}\n\n"
                 f"Knowledge-graph facts:\n{facts_block}\n\n"
                 f"Relevant images (described):\n{img_block}\n\n"
                 f"Question: {query}"
             )},
        ],
    )
    return resp.choices[0].message.content


def ask_full(query, text_index, chunks, sources, graph,
             img_index, img_names, captions, k=3):
    text_retrieved = retrieve(query, text_index, chunks, sources, k=k)
    graph_facts = graph_facts_for_query(query, graph)
    image_retrieved = retrieve_images(query, img_index, img_names, captions, k=1)
    answer = generate_full(query, text_retrieved, graph_facts, image_retrieved)
    return answer, text_retrieved, graph_facts, image_retrieved


# ---------- demo ----------
if __name__ == "__main__":
    text_index, chunks, sources = build_index()
    graph = build_graph(build_triples())
    img_index, img_names, captions = build_image_index()

    print("\nFull enhanced RAG (text + KG + multimodal). Type 'quit' to exit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit", "q", ""):
            break
        answer, text_retrieved, facts, image_retrieved = ask_full(
            question, text_index, chunks, sources, graph, img_index, img_names, captions
        )
        print("\nGraph facts used:")
        for fact in facts:
            print(f"  - {fact}")
        print("Image retrieved:")
        for score, name, caption in image_retrieved:
            print(f"  {score:.3f} : {name}")
        print(f"\nAnswer: {answer}\n" + "-" * 60)