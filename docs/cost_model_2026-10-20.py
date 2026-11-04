"""Cost model for the configs as of 2026-10-20 (USD per 1M tokens, OpenRouter).

Seven generators, every arm on every generator, the oracle-image and +both
arms, and the 132-item caption-flagged SPIQA set. Prices are those frozen in
configs/pricing_2026-10-20.json. Token means follow the 2026-09-10 model, with
5 passages per text prompt instead of 3.

Hidden reasoning is the unknown. Three generators (both Gemini models and
Qwen 3.8 Max) and the GPT-6.1 Sol and Opus 5.5 judges think before answering,
and OpenRouter bills those tokens as output. The listings do not say how many,
so the model prints a scenario without them and one with a fixed allowance;
the --limit 5 pilot measures the real figure.
"""

P = {  # (input, output)
    "gpt4o-mini": (0.15, 0.60),
    "gemini-flash-lite": (0.25, 1.50),
    "llama4-scout": (0.10, 0.30),
    "llama4-maverick": (0.35, 1.00),
    "qwen3.7-flash": (0.03, 0.13),
    "gemini-3.8-flash": (0.825, 4.125),
    "qwen3.8-max": (2.00, 6.00),
    "gpt-6.1-sol": (2.00, 10.00),
    "claude-opus-5.5": (4.00, 20.00),
    "deepseek-v4-pro": (0.66, 1.98),
    "grok-4.7": (2.00, 6.00),
}
GENERATORS = [
    "gpt4o-mini",
    "gemini-flash-lite",
    "llama4-scout",
    "llama4-maverick",
    "qwen3.7-flash",
    "gemini-3.8-flash",
    "qwen3.8-max",
]
THINKING_GENERATORS = {"gemini-flash-lite", "gemini-3.8-flash", "qwen3.8-max"}
THINKING_JUDGES = {"gpt-6.1-sol", "claude-opus-5.5"}

TEXT_IN, TEXT_OUT = 600, 15
PIX_IN, PIX_OUT = 2000, 30  # 5 passages plus one crop
ACC_IN, ACC_OUT = 150, 40  # one-sentence justification, then the digit
FAITH_IN, FAITH_OUT = 800, 40
VF_IN, VF_OUT = 2000, 40

# name: (questions, outputs per question per generator, pixel outputs among
# them, judge). Visual sets run twice (pixels and captions); pixels reach
# the generator only in the pixel run, on +multimodal, +both, and oracle-image.
SETS = {
    "hotpotqa bridge": (100, 13, 0, "deepseek-v4-pro"),
    "hotpotqa comparison": (100, 13, 0, "deepseek-v4-pro"),
    "spiqa cross": (50, 6, 0, "deepseek-v4-pro"),
    "spiqa native primary": (150, 12, 3, "gpt-6.1-sol"),
    "spiqa native flagged": (132, 12, 3, "gpt-6.1-sol"),
    "publaynet figures": (35, 10, 3, "gpt-6.1-sol"),
}
REPEAT_SHARE = 0.1 * 2  # 10% subset, two extra runs, primary sets only


def cost(model, n, tokens_in, tokens_out):
    price_in, price_out = P[model]
    return n * (tokens_in * price_in + tokens_out * price_out) / 1e6


def estimate(gen_reasoning, judge_reasoning):
    lines = {}
    generation = judging = 0.0
    sol_outputs = image_outputs = 0
    for name, (questions, per_q, pixel_per_q, judge) in SETS.items():
        repeat = 1 + (
            REPEAT_SHARE if name.startswith(("hotpotqa", "spiqa native p")) else 0
        )
        text_n = questions * (per_q - pixel_per_q) * repeat
        pixel_n = questions * pixel_per_q * repeat
        for model in GENERATORS:
            hidden = gen_reasoning if model in THINKING_GENERATORS else 0
            generation += cost(model, text_n, TEXT_IN, TEXT_OUT + hidden)
            generation += cost(model, pixel_n, PIX_IN, PIX_OUT + hidden)
        outputs = (text_n + pixel_n) * len(GENERATORS)
        image_n = pixel_n * len(GENERATORS)
        hidden = judge_reasoning if judge in THINKING_JUDGES else 0
        judging += cost(judge, outputs, ACC_IN, ACC_OUT + hidden)
        judging += cost(judge, outputs - image_n, FAITH_IN, FAITH_OUT + hidden)
        judging += cost(judge, image_n, VF_IN, VF_OUT + hidden)
        if judge == "gpt-6.1-sol":
            sol_outputs += outputs
            image_outputs += image_n
    lines["generation, seven generators"] = generation
    lines["primary judges (DeepSeek text, GPT-6.1 Sol figures)"] = judging
    lines["Grok 4.7 second vision judge, 20% of image arms"] = cost(
        "grok-4.7", 0.2 * image_outputs, VF_IN + ACC_IN, 2 * ACC_OUT
    )
    lines["Opus 5.5 adjudication, 10% of Sol-judged outputs"] = cost(
        "claude-opus-5.5", 0.1 * sol_outputs, 800, 60 + judge_reasoning
    )
    lines["sufficiency checks (3,000, Sol)"] = cost(
        "gpt-6.1-sol", 3000, FAITH_IN, FAITH_OUT + judge_reasoning
    )
    lines["review, triple audit, calibration, indexing, extraction"] = 11.0
    return lines


for label, gen_hidden, judge_hidden in (
    ("no hidden reasoning", 0, 0),
    ("400 hidden tokens per thinking generator call, 300 per judge call", 400, 300),
):
    lines = estimate(gen_hidden, judge_hidden)
    print(f"\n{label}")
    for item, value in lines.items():
        print(f"  {item:<58} {value:7.0f}")
    print(f"  {'total':<58} {sum(lines.values()):7.0f}")
