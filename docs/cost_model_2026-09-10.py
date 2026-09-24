"""Lean-design cost model with OpenRouter prices fetched 2026-09-10 (USD per 1M tokens)."""

P = {  # (input, output)
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "meta-llama/llama-4-scout": (0.10, 0.30),
    "meta-llama/llama-4-maverick": (0.20, 0.696),
    "qwen/qwen3.7-flash": (0.03, 0.13),
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-sonnet-5:batch": (1.0, 5.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "anthropic/claude-opus-5:batch": (2.5, 12.5),
    "deepseek/deepseek-v4-pro-0813": (0.579, 1.738),
    "x-ai/grok-4.3": (1.25, 2.5),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "openai/text-embedding-3-small": (0.02, 0.0),
}


def cost(model, n, tin, tout):
    i, o = P[model]
    return n * (tin * i + tout * o) / 1e6


GENS = [
    "openai/gpt-4o-mini",
    "google/gemini-3.1-flash-lite",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3.7-flash",
]
STRONG2 = ["openai/gpt-4o-mini", "meta-llama/llama-4-maverick"]  # weak + strong pair

# question-arm counts per set: (n_questions, arms_all5, arms_two)
sets = {
    "hotpotqa (text)": (
        200,
        5 + 2,
        4 + 2,
    ),  # primary 5 + per-gen controls 2 ; secondary 4 + shared controls 2
    "spiqa cross (text)": (50, 5 + 2, 4 + 2),
    "spiqa native (pixel)": (150, 6, 1),
    "publaynet figures (pixel)": (100, 4, 0),
    "text mechanism (text)": (200, 0, 3),
}
TEXT_IN, TEXT_OUT = 400, 15
PIX_IN, PIX_OUT = 1500, 30

gen_cost = 0.0
outputs = {"text": 0, "pixel": 0}
for name, (nq, a5, a2) in sets.items():
    pixel = "pixel" in name
    tin, tout = (PIX_IN, PIX_OUT) if pixel else (TEXT_IN, TEXT_OUT)
    for g in GENS:
        gen_cost += cost(g, nq * a5, tin, tout)
    for g in STRONG2:
        gen_cost += cost(g, nq * a2, tin, tout)
    n = nq * (a5 * len(GENS) + a2 * len(STRONG2))
    outputs["pixel" if pixel else "text"] += n
# repeats: 10% of primary arms, two extra runs, all five generators
rep_text = int(0.1 * (200 + 50) * 5 * 5 * 2)
rep_pix = int(0.1 * 150 * 6 * 5 * 2 + 0.1 * 100 * 4 * 5 * 2)
for g in GENS:
    gen_cost += cost(g, rep_text / 5, TEXT_IN, TEXT_OUT) + cost(
        g, rep_pix / 5, PIX_IN, PIX_OUT
    )
outputs["text"] += rep_text
outputs["pixel"] += rep_pix
print("outputs", outputs, "total", sum(outputs.values()))
print(f"generation                          {gen_cost:6.2f}")

ACC_IN, ACC_OUT = 150, 40  # accuracy judge with one-sentence justification
FAITH_IN, FAITH_OUT = 600, 40  # text faithfulness
VF_IN, VF_OUT = 1500, 40  # vision faithfulness (image attached)
n_text, n_pix = outputs["text"], outputs["pixel"]
noncontrol_text = int(n_text * 0.7)

ds = cost("deepseek/deepseek-v4-pro-0813", n_text + n_pix, ACC_IN, ACC_OUT) + cost(
    "deepseek/deepseek-v4-pro-0813", noncontrol_text, FAITH_IN, FAITH_OUT
)
print(f"deepseek v4-pro second judge        {ds:6.2f}")


# Under-60 configuration (2026-09-10): Sonnet judges the visual sets only; the text
# sets are judged in full by DeepSeek V4 Pro with EM/F1 as the headline. Vision
# faithfulness is judged only on arms that supplied an image (about half of the
# pixel outputs); text-only arms in the visual sets use the cheap text prompt.
IMAGE_ARM_SHARE = 0.5


def sonnet(model):
    s = cost(model, n_pix, ACC_IN, ACC_OUT)
    s += cost(model, int(n_pix * IMAGE_ARM_SHARE), VF_IN, VF_OUT)
    s += cost(model, int(n_pix * (1 - IMAGE_ARM_SHARE) * 0.8), FAITH_IN, FAITH_OUT)
    return s


print(f"sonnet 5 primary judge (standard)   {sonnet('anthropic/claude-sonnet-5'):6.2f}")
print(
    f"sonnet 5 primary judge (batch)      {sonnet('anthropic/claude-sonnet-5:batch'):6.2f}"
)

grok_n = int(0.2 * n_pix * IMAGE_ARM_SHARE)
grok = cost("x-ai/grok-4.3", grok_n, VF_IN, VF_OUT) + cost(
    "x-ai/grok-4.3", int(0.2 * n_pix), ACC_IN, ACC_OUT
)
print(f"grok 4.3 second vision judge (20%)  {grok:6.2f}")

judged_by_sonnet = n_pix
opus = cost("anthropic/claude-opus-5", int(0.1 * judged_by_sonnet), 600, 60)
opus_b = cost("anthropic/claude-opus-5:batch", int(0.1 * judged_by_sonnet), 600, 60)
print(f"opus 5 adjudication (10% disagree)  {opus:6.2f}  batch {opus_b:6.2f}")

suff = cost("anthropic/claude-sonnet-5", 3000, FAITH_IN, FAITH_OUT)
print(f"sufficiency checks (3000)           {suff:6.2f}")

review = cost("anthropic/claude-sonnet-5", 265, 1500, 120) + cost(
    "x-ai/grok-4.3", 170, 1500, 120
)
triples = cost("anthropic/claude-sonnet-5", 300, 500, 80) + cost(
    "deepseek/deepseek-v4-pro-0813", 100, 500, 80
)
calib = cost("anthropic/claude-sonnet-5", 1400, ACC_IN, ACC_OUT) + cost(
    "deepseek/deepseek-v4-pro-0813", 1400, ACC_IN, ACC_OUT
)
print(f"question review + triple audit + calibration {review + triples + calib:6.2f}")

# HotpotQA corpus rebuilt 2026-09-10 from a 600-question pool: 10,108 chunks, of which
# 4,974 have no cached triples yet (our extractor, one call each).
HOTPOT_CHUNKS, NEW_CHUNKS = 10108, 4974
hippo = cost("openai/gpt-4o-mini", HOTPOT_CHUNKS * 2, 800, 200) + cost(
    "openai/text-embedding-3-small", HOTPOT_CHUNKS, 300, 0
)
extract = cost("openai/gpt-4o-mini", NEW_CHUNKS, 800, 200)
print(f"triple extraction, new hotpotqa chunks {extract:6.2f}")
print(f"hipporag indexing hotpotqa          {hippo:6.2f}")
newq = cost("anthropic/claude-haiku-4.5", 200, 1500, 120)
print(f"new publaynet figure questions      {newq:6.2f}")

std = (
    gen_cost
    + ds
    + sonnet("anthropic/claude-sonnet-5")
    + grok
    + opus
    + suff
    + review
    + triples
    + calib
    + hippo
    + newq
)
bat = (
    gen_cost
    + ds
    + sonnet("anthropic/claude-sonnet-5:batch")
    + grok
    + opus_b
    + suff
    + review
    + triples
    + calib
    + hippo
    + newq
)
print(f"\nTOTAL standard {std:6.2f}   TOTAL with batch variants {bat:6.2f}")
