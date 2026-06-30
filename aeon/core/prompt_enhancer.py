"""LLM-driven prompt enhancement for the image/video generators.

The local generators reward verbose, concrete prompts far more than terse ones:
FLUX/Qwen images want composition/lighting/lens detail, and LTX/10Eros video has
almost no self-reasoning -- any unstated motion, camera move, or detail simply
does not happen. This rewrites a short prompt into the dialect each generator
wants, using the single (uncensored, local) agent model, so explicit intent is
preserved -- a censored rewriter would silently sanitize it.

Fail loud, never fall back: if enhancement is requested but the local model is
unreachable or errors, the exception propagates rather than silently passing the
un-enhanced prompt through (which would be a silent degradation). Returning the
prompt unchanged is reserved for the legitimate no-op cases: enhancement turned
off (force=False) or an already-detailed prompt.
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

    No-op (returns raw unchanged) only for the legitimate cases: empty prompt,
    force=False, or an already-detailed prompt under auto mode. If enhancement is
    actually attempted and the local model errors, the exception propagates --
    there is no fallback model and no silent pass-through.
    """
    raw = (raw or "").strip()
    if not raw or force is False:
        return raw
    if force is None and len(raw.split()) >= _PASSTHROUGH_WORDS:
        return raw
    if llm_client is None:
        raise ValueError("enhance_prompt: enhancement requested but no llm_client was provided.")

    system = _SYSTEM.get(media_type, _SYSTEM["image"])
    resp = llm_client.client.chat.completions.create(
        # Send the served model id (api_model), the same id the main agent loop
        # uses -- the display name ('model') 404s against vLLM's served name.
        model=llm_client.api_model,
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
