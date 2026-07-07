"""Startup vision self-test — the hard gate that makes a broken multimodal model
fail LOUD instead of silently confabulating.

A model marked ``multimodal=True`` in ``model_catalog.py`` is trusted by the
browser loop (screenshots are attached to the agent's own prompt) and by
``analyze_image``. Failure modes, only the first two of which surface on their
own:

  1. vLLM won't serve the vision architecture   -> endpoint unreachable / 5xx
  2. the model is actually text-only             -> HTTP 400 rejecting the image
  3. the model "sees" but MISREADS text          -> HTTP 200 with wrong content

Case 3 is the dangerous one and the one we actually hit on the abliterated
Qwen3.6 FP8 build: the vision tower is intact (BF16, un-quantized) but the
uniformly-abliterated language layers that interpret vision tokens read a crisp
'RP9PCV' back as 'R171' — coarse shape seen, fine detail garbled. That is NOT a
quant-damaged tower and NOT a probe artifact; it is degraded visual *reading*.

This probe renders a random nonce into a legible, screenshot-like image and
requires the model to read it back. It samples several times and requires a
majority to guard against a one-off decode slip. On real failure it saves every
probe image to disk and raises ``VisionSelfTestError`` with the paths and the raw
readings, so the caller aborts the agent with everything needed to diagnose,
rather than letting a half-blind model drive the browser.
"""
from __future__ import annotations

import io
import os
import base64
import logging
import secrets
import tempfile

import requests

logger = logging.getLogger("aeon")

# Unambiguous OCR charset: no 0/O, 1/I/L, 2/Z, 5/S, 8/B, 6/G, D/Q collisions.
_NONCE_ALPHABET = "ACEFHJKMNPRTUVWXY34679"
_NONCE_LEN = 6

# Sampling: render N codes, pass as soon as PASS_NEED read back correctly. A
# healthy model passes on the first two calls; a broken one gets all N tries.
_SAMPLES = 3
_PASS_NEED = 2

_SAVE_DIR = os.path.join(tempfile.gettempdir(), "aeon_vision_selftest")


class VisionSelfTestError(RuntimeError):
    """The declared-multimodal model failed to prove it can actually read.

    ``hint`` is a human-facing remediation string; ``images`` are on-disk paths
    to the probe images that were sent, for the operator to inspect.
    """

    def __init__(self, message: str, hint: str = "", images=None):
        super().__init__(message)
        self.hint = hint
        self.images = images or []


def _make_nonce() -> str:
    return "".join(secrets.choice(_NONCE_ALPHABET) for _ in range(_NONCE_LEN))


def _load_font(size: int):
    """Best-effort legible font at ``size`` px (never raises)."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        pass
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _centered(draw, text, font, y, W):
    try:
        l, _, r, _ = draw.textbbox((0, 0), text, font=font)
        x = (W - (r - l)) / 2 - l
    except Exception:
        x = 40
    draw.text((x, y), text, fill=(17, 17, 17), font=font)


def _render_probe(nonce: str) -> bytes:
    """A screenshot-like verification card with the nonce in normal-sized text —
    representative of the web text the browser agent must read (not one giant
    OOD glyph string).

    CRITICAL: render at the SAME resolution the browser actually sends. The
    browser downsizes screenshots to VISION_MAX_DIM=1920 (aeon/core/llm.py), and
    Qwen3-VL-class models resolve fine text only when they get enough vision
    tokens: a 768 px probe made this model misread '...4RA' as 'MAMA', but at
    1600 px it reads 6/6. A too-small probe FALSE-FAILS a model that browses
    fine. High contrast, JPEG. Raises on PIL failure."""
    from PIL import Image, ImageDraw
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), (244, 246, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([W * 0.08, H * 0.18, W * 0.92, H * 0.82],
                   fill=(255, 255, 255), outline=(210, 214, 220), width=4)
    _centered(draw, "Verification code", _load_font(int(H * 0.06)), int(H * 0.28), W)
    _centered(draw, nonce, _load_font(int(H * 0.16)), int(H * 0.44), W)
    _centered(draw, "Enter the code above to continue.", _load_font(int(H * 0.045)), int(H * 0.72), W)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _save_probe(jpeg: bytes, idx: int, nonce: str) -> str:
    try:
        os.makedirs(_SAVE_DIR, exist_ok=True)
        p = os.path.join(_SAVE_DIR, f"probe_{idx}_{nonce}.jpg")
        with open(p, "wb") as f:
            f.write(jpeg)
        return p
    except Exception:
        return "<unsaved>"


def _ask_vision(url: str, model: str, jpeg: bytes, timeout: int):
    """POST one image, return (answer_text, None) or (None, VisionSelfTestError).

    Transport/HTTP failures are terminal (raise-worthy) and returned as the error;
    a 200 with content returns the text for the caller to score.
    """
    b64 = base64.b64encode(jpeg).decode("utf-8")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",
                 "text": ("This is a screenshot. Read the verification code shown "
                          "in large text and reply with ONLY that code (letters and "
                          "digits), nothing else.")},
            ],
        }],
        # Room for the model's chain-of-thought to complete and state the code.
        # A tight cap (e.g. 64) truncates reasoning mid-stream and FALSE-FAILS a
        # thinking model that would have read it correctly. We match the nonce
        # anywhere in the full output, so the reasoning stating it counts.
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return None, VisionSelfTestError(
            f"vision endpoint unreachable at {url}: {type(e).__name__}: {e}",
            hint=("The multimodal model is not serving on its chat endpoint. "
                  "Check that the model launched and that this vLLM build "
                  "supports the checkpoint's vision architecture."))
    if resp.status_code != 200:
        body = (resp.text or "")[:600]
        low = body.lower()
        if resp.status_code == 400 and ("image" in low or "multimodal" in low or "vision" in low):
            return None, VisionSelfTestError(
                f"server REJECTED the image (HTTP 400): {body}",
                hint=("The served model is TEXT-ONLY — vLLM loaded it without a "
                      "vision tower. Either the checkpoint isn't a multimodal "
                      "*ForConditionalGeneration, or this vLLM version doesn't "
                      "register it. Fix the serving arch, or set multimodal=False "
                      "for this entry in aeon/core/model_catalog.py."))
        return None, VisionSelfTestError(
            f"vision endpoint returned HTTP {resp.status_code}: {body}",
            hint="The model errored on an image request; inspect the server log above.")
    try:
        answer = resp.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        return None, VisionSelfTestError(
            f"unexpected response format from vision endpoint: {type(e).__name__}: {e}; "
            f"raw={resp.text[:400]}",
            hint="The endpoint answered but not in OpenAI chat format; check the server.")
    return answer, None


def _reads(answer: str, nonce: str) -> bool:
    return nonce in "".join(ch for ch in answer.upper() if ch.isalnum())


# Failure hint for the case that actually bites: the model sees the image and
# answers, but reads the text wrong. Reflects the real Qwen3.6-abliterated finding.
_MISREAD_HINT = (
    "The model SEES the image but cannot READ text reliably. The vision tower is "
    "fine (it produced coherent output); the damage is in the language layers' "
    "ability to interpret vision tokens — the hallmark of an abliteration that ran "
    "uniformly across all attention/MLP layers (AEON-7 lineage). Do NOT trust this "
    "model for browsing. Use a vision-PRESERVING abliteration or the official "
    "Qwen3.6 vision weights, or set multimodal=False for this entry in "
    "aeon/core/model_catalog.py. Inspect the saved probe images to confirm they are "
    "legible (they are crisp, normal-sized text)."
)


def run_vision_self_test(base_url: str, model: str, timeout: int = 120) -> str:
    """Prove the served model can READ an image, or raise VisionSelfTestError.

    Returns the last correctly-read nonce on success, or "" if the probe itself
    could not run (our bug, not the model's — logged and skipped, never aborts).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    passes = 0
    records = []   # (nonce, answer, ok)
    images = []
    for i in range(_SAMPLES):
        try:
            nonce = _make_nonce()
            jpeg = _render_probe(nonce)
        except Exception as e:
            # Probe infrastructure failure (e.g. no PIL/font) is OURS, not the
            # model's — warn and skip so it never bricks startup for a model
            # whose vision may be fine.
            logger.warning("Vision self-test SKIPPED — could not build probe image: %s", e)
            print(f"\033[93m[VISION SELF-TEST] Skipped (probe build failed: {e}). "
                  f"Vision is UNVERIFIED this session.\033[0m")
            return ""

        images.append(_save_probe(jpeg, i, nonce))
        answer, err = _ask_vision(url, model, jpeg, timeout)
        if err is not None:
            err.images = images
            raise err

        ok = _reads(answer, nonce)
        records.append((nonce, answer.strip(), ok))
        if ok:
            passes += 1
            if passes >= _PASS_NEED:
                return nonce
        # Can the remaining samples still reach the bar? If not, stop early.
        if passes + (_SAMPLES - i - 1) < _PASS_NEED:
            break

    detail = "; ".join(
        f"[{'OK' if ok else 'MISREAD'}] expected {n!r} -> got {a[:80]!r}"
        for n, a, ok in records
    )
    raise VisionSelfTestError(
        f"model could not read the probe: {passes}/{len(records)} correct "
        f"(need {_PASS_NEED}). {detail}",
        hint=_MISREAD_HINT,
        images=images,
    )
