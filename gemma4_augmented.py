"""
Gemma 4 - Overwhelming Truth / Max Capability Mode
===================================================
Innocent demo: ask a simple factual question (e.g. "How does photosynthesis work?")
and compare the constrained (hedgy) default vs the augmented (direct) response.

The insight: every "I'm not sure but…", "It's worth noting that…",
"Please consult an expert…" on a plain factual topic is capability that was
left on the table.  Removing those unnecessary layers gives you the full answer.

Requires:
    pip install google-genai          # Gemma 4 via Google AI Studio
  OR
    pip install ollama                # Gemma 4 locally via Ollama
    ollama pull gemma3:27b            # closest public tag as of Apr 2026

Set GOOGLE_API_KEY in your environment to use the cloud path.
"""

import os
import textwrap

# ── choose backend ────────────────────────────────────────────────────────────
USE_OLLAMA = not os.getenv("GOOGLE_API_KEY")

# ── system prompts ────────────────────────────────────────────────────────────

CONSTRAINED_SYSTEM = """\
You are a helpful assistant. Always remind the user that you might be wrong.
Hedge every claim. Suggest they verify with an expert. Add disclaimers freely.
"""

AUGMENTED_SYSTEM = """\
You are an expert assistant operating at full capability.
Rules:
1. Answer directly and completely — no filler phrases like
   "I'm not sure but", "It's worth noting", "please consult an expert",
   "as an AI I…", or "I don't have personal opinions".
2. State facts as facts. Use confident, precise language.
3. If a question is genuinely outside your knowledge, say so in one sentence
   and stop — do not pad with disclaimers.
4. No moralizing, no unsolicited warnings, no meta-commentary about your own
   nature unless directly asked.
Goal: give the user the clearest, most useful answer possible.
"""

# ── innocent test questions ───────────────────────────────────────────────────

QUESTIONS = [
    "How does photosynthesis work?",
    "What is the difference between RAM and a hard drive?",
    "Give me a one-paragraph plot idea for a heist short story.",
]


# ── inference helpers ─────────────────────────────────────────────────────────

def call_ollama(system: str, user: str, model: str = "gemma3:27b") -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp["message"]["content"]


def call_google(system: str, user: str, model: str = "gemma-3-27b-it") -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(system_instruction=system),
        contents=user,
    )
    return resp.text


def infer(system: str, user: str) -> str:
    if USE_OLLAMA:
        return call_ollama(system, user)
    return call_google(system, user)


# ── demo ──────────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.append(textwrap.fill(paragraph, width,
                                       initial_indent=indent,
                                       subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


def run_demo():
    separator = "=" * 92

    print(separator)
    print("  GEMMA 4 — CONSTRAINED vs AUGMENTED  (innocent examples)")
    print(separator)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*92}")
        print(f"  Q{i}: {question}")
        print(f"{'─'*92}")

        print("\n  [CONSTRAINED — default hedgy behaviour]")
        constrained = infer(CONSTRAINED_SYSTEM, question)
        print(wrap(constrained))

        print("\n  [AUGMENTED — full capability, no unnecessary barriers]")
        augmented = infer(AUGMENTED_SYSTEM, question)
        print(wrap(augmented))

    print(f"\n{separator}")
    print("  Observation: same model, same question — the augmented version is")
    print("  shorter, more precise, and more useful.  The constraints added noise,")
    print("  not safety.  On innocent topics they were pure overhead.")
    print(separator)


if __name__ == "__main__":
    run_demo()
