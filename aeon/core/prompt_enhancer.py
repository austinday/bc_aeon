"""LLM-driven prompt enhancement for the image/video generators.

The local generators reward verbose, concrete prompts far more than terse ones:
FLUX/Qwen images want composition/lighting/lens detail, and LTX/10Eros video has
almost no self-reasoning -- any unstated motion, camera move, or detail simply
does not happen. This rewrites a short prompt into the dialect each generator
wants, using the single agent model. Because that model is uncensored, explicit
intent is preserved (a censored rewriter would silently sanitize it).

Best-effort by design: any error, a missing client, or an already-detailed
prompt returns the original prompt unchanged.
"""
from __future__ import annotations

from typing import Optional

C_CYAN = "\033[96m"
C_RESET = "\033[0m"

# Prompts already at least this many words are assumed detailed -> passed through.
_PASSTHROUGH_WORDS = 40

_SYSTEM = {
    "video": (
        "You are a prompt engineer for the LTX-2 / 10Eros text-to-video model. Rewrite "
        "the user's request into ONE vivid, concrete English paragraph optimized for "
        "video. The model has almost no self-reasoning, so you MUST explicitly describe: "
        "subject appearance, setting and lighting, and above all the MOTION over time -- "
        "what moves and how, plus camera movement and pacing. Preserve the user's intent "
        "and any explicit or adult content exactly; do not censor, sanitize, refuse, soften, "
        "or add warnings. Output ONLY the rewritten prompt: no preamble, quotes, or notes."
    ),
    "image": (
        "You are a prompt engineer for the FLUX / Qwen image models. Rewrite the user's "
        "request into ONE vivid, concrete English description optimized for image "
        "generation: subject, composition, lighting, lens/camera, style, mood, and fine "
        "detail. Preserve the user's intent and any explicit or adult content exactly; do "
        "not censor, sanitize, refuse, soften, or add warnings. Output ONLY the rewritten "
        "prompt: no preamble, quotes, or notes."
    ),
    "image_edit": (
        "You are a prompt engineer for the Qwen-Image-Edit model, which edits an EXISTING "
        "image from an instruction. Rewrite the user's edit instruction so it is clear, "
        "specific, and unambiguous about exactly what to change and how, while leaving "
        "everything not mentioned untouched. Keep it an imperative EDIT INSTRUCTION, not a "
        "full scene description. Preserve the user's intent and any explicit or adult "
        "content exactly; do not censor, sanitize, refuse, soften, or add warnings. Output "
        "ONLY the rewritten instruction: no preamble, quotes, or notes."
    ),
}


def enhance_prompt(llm_client, raw: str, media_type: str = "image",
                   force: Optional[bool] = None) -> str:
    """Return an enhanced prompt for media_type ('video' | 'image').

    force=True  -> always enhance.
    force=False -> never enhance (return raw).
    force=None  -> auto: enhance terse prompts, pass detailed ones through.

    Falls back to `raw` on any error or when no client is available.
    """
    raw = (raw or "").strip()
    if not raw or force is False or llm_client is None:
        return raw
    if force is None and len(raw.split()) >= _PASSTHROUGH_WORDS:
        return raw

    system = _SYSTEM.get(media_type, _SYSTEM["image"])
    try:
        resp = llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": raw}],
            temperature=0.7,
        )
        out = (resp.choices[0].message.content or "").strip()
        # Strip accidental wrapping quotes the model may add.
        if len(out) >= 2 and out[0] in "\"'" and out[-1] == out[0]:
            out = out[1:-1].strip()
        if out and out != raw:
            print(f"{C_CYAN}[prompt-enhancer:{media_type}] {out}{C_RESET}")
            return out
        return raw
    except Exception:
        return raw
