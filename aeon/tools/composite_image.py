import math
import os
import stat
import warnings

from PIL import Image

from .base import BaseTool
from ..core.paths import resolve_output_dir


# ``Image.open`` is lazy, but ``convert``/``resize`` materialize the complete
# decoded raster in Aeon's process.  Keep those allocations explicitly bounded;
# Pillow's much larger process-global decompression-bomb default is not a
# resource contract for this tool.
MAX_BASE_IMAGE_BYTES = 64 * 1024 * 1024
MAX_OVERLAY_IMAGE_BYTES = 32 * 1024 * 1024
MAX_BASE_PIXELS = 36_000_000
MAX_OVERLAY_PIXELS = 16_000_000
MAX_RESIZED_OVERLAY_PIXELS = 20_000_000
MAX_IMAGE_DIMENSION = 16_384
MAX_DECODED_RGBA_BYTES = 256 * 1024 * 1024


class _CompositeLimitError(ValueError):
    """Raised before an unbounded image decode or resize is attempted."""


def _regular_file_size(path: str, *, label: str, max_bytes: int) -> int:
    try:
        file_stat = os.stat(path)
    except OSError as exc:
        raise _CompositeLimitError(f"could not inspect {label}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise _CompositeLimitError(f"{label} must be a regular file")
    if file_stat.st_size <= 0:
        raise _CompositeLimitError(f"{label} is empty")
    if file_stat.st_size > max_bytes:
        raise _CompositeLimitError(
            f"{label} is {file_stat.st_size:,} bytes; limit is {max_bytes:,} bytes"
        )
    return int(file_stat.st_size)


def _validated_dimensions(image: Image.Image, *, label: str, max_pixels: int) -> tuple[int, int]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise _CompositeLimitError(f"{label} has invalid dimensions {width}x{height}")
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise _CompositeLimitError(
            f"{label} dimensions {width}x{height} exceed the "
            f"{MAX_IMAGE_DIMENSION:,}-pixel side limit"
        )
    pixels = width * height
    if pixels > max_pixels:
        raise _CompositeLimitError(
            f"{label} has {pixels:,} pixels; limit is {max_pixels:,} pixels"
        )
    return width, height


class CompositeImageTool(BaseTool):
    """Deterministically paste one image (a logo/graphic, ideally a transparent
    PNG) onto another at an exact position/size/opacity. Unlike edit_image (which
    is diffusion and REDRAWS what it touches, mangling logo text/shapes), this is
    pixel-exact — the logo comes out crisp and on-brand. The final step of an ad."""

    def __init__(self):
        super().__init__(
            name="composite_image",
            description=(
                "Stamp a logo/graphic onto a base image with PIXEL-EXACT placement "
                "(alpha compositing, no AI redraw). Use this to put a brand logo, "
                "badge, or product cutout onto a generated/edited ad background so it "
                "stays crisp and unaltered — edit_image would distort it. For a clean "
                "result the overlay should be a transparent PNG.\n"
                "Schema:\n"
                "  base_path (str, required): the background image to paste onto.\n"
                "  overlay_path (str, required): the logo/graphic to place (transparent PNG best).\n"
                "  output_dir (str, REQUIRED): the DIRECTORY to save into; the file is auto-named "
                "'<base>_composited.png' and its full path is returned. Use '.' for the current workspace.\n"
                "  position (str, optional, default 'bottom-right'): one of top-left, top-right, "
                "bottom-left, bottom-right, center, top, bottom, left, right; OR exact pixels 'x,y'.\n"
                "  scale (float, optional, default 0.2): overlay width as a FRACTION of the base "
                "width when <=1 (0.2 = 20% of base width), or an explicit pixel width when >1.\n"
                "  opacity (float, optional, default 1.0): 0..1 (e.g. 0.5 for a watermark).\n"
                "  margin (int, optional): pixels from the edge for corner/edge positions "
                "(default = 3% of base width).\n"
                "Example: {\"tool_name\": \"composite_image\", \"parameters\": {\"base_path\": "
                "\"ad_bg.png\", \"overlay_path\": \"logo.png\", \"output_dir\": \".\", "
                "\"position\": \"top-right\", \"scale\": 0.18}}"
            ),
            directives=[],
        )

    @staticmethod
    def _num(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _position(self, position, base_size, ov_size, margin):
        bw, bh = base_size
        ow, oh = ov_size
        p = str(position or "bottom-right").strip().lower().replace("_", "-").replace(" ", "-")
        if "," in p:  # explicit "x,y" pixel coordinates
            try:
                parts = [int(float(t)) for t in p.split(",")[:2]]
                return parts[0], parts[1]
            except (ValueError, IndexError):
                pass
        horiz = "left" if "left" in p else "right" if "right" in p else "center"
        vert = "top" if "top" in p else "bottom" if "bottom" in p else "center"
        x = margin if horiz == "left" else (bw - ow - margin) if horiz == "right" else (bw - ow) // 2
        y = margin if vert == "top" else (bh - oh - margin) if vert == "bottom" else (bh - oh) // 2
        return x, y

    def execute(self, base_path: str, overlay_path: str, output_dir: str = None,
                position: str = "bottom-right", scale=0.2, opacity=1.0, margin=None) -> str:
        if not base_path:
            return "Error: 'base_path' is required."
        if not overlay_path:
            return "Error: 'overlay_path' (the logo/graphic to place) is required."
        if not output_dir or not str(output_dir).strip():
            return "Error: 'output_dir' is required — the directory to save the composited image in."
        base_abs, ov_abs = os.path.abspath(base_path), os.path.abspath(overlay_path)
        if not os.path.exists(base_abs):
            return f"Error: base image not found at {base_abs}"
        if not os.path.exists(ov_abs):
            return f"Error: overlay image not found at {ov_abs}"

        try:
            _regular_file_size(
                base_abs, label="base image", max_bytes=MAX_BASE_IMAGE_BYTES
            )
            _regular_file_size(
                ov_abs, label="overlay image", max_bytes=MAX_OVERLAY_IMAGE_BYTES
            )
            scale = self._num(scale, 0.2)
            if not math.isfinite(scale) or scale <= 0:
                scale = 0.2

            # Pillow only reads headers at ``open``.  Validate both source rasters
            # and the planned resize/peak allocation before either ``convert``
            # materializes pixel data.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(base_abs) as base_source, Image.open(ov_abs) as overlay_source:
                    base_w, base_h = _validated_dimensions(
                        base_source, label="base image", max_pixels=MAX_BASE_PIXELS
                    )
                    overlay_w, overlay_h = _validated_dimensions(
                        overlay_source,
                        label="overlay image",
                        max_pixels=MAX_OVERLAY_PIXELS,
                    )

                    # Resize overlay: scale <=1 is a fraction of base width; >1
                    # is a pixel width.
                    target_w = (
                        max(1, int(base_w * scale))
                        if scale <= 1.0
                        else min(int(scale), base_w)
                    )
                    target_h = max(1, round(overlay_h * (target_w / overlay_w)))
                    if target_w > MAX_IMAGE_DIMENSION or target_h > MAX_IMAGE_DIMENSION:
                        raise _CompositeLimitError(
                            f"resized overlay dimensions {target_w}x{target_h} exceed the "
                            f"{MAX_IMAGE_DIMENSION:,}-pixel side limit"
                        )
                    resized_pixels = target_w * target_h
                    if resized_pixels > MAX_RESIZED_OVERLAY_PIXELS:
                        raise _CompositeLimitError(
                            f"resized overlay has {resized_pixels:,} pixels; limit is "
                            f"{MAX_RESIZED_OVERLAY_PIXELS:,} pixels"
                        )
                    peak_decoded_bytes = (
                        base_w * base_h + overlay_w * overlay_h + resized_pixels
                    ) * 4
                    if peak_decoded_bytes > MAX_DECODED_RGBA_BYTES:
                        raise _CompositeLimitError(
                            f"compositing requires at least {peak_decoded_bytes:,} decoded "
                            f"RGBA bytes; limit is {MAX_DECODED_RGBA_BYTES:,} bytes"
                        )

                    base = base_source.convert("RGBA")
                    overlay = overlay_source.convert("RGBA")

            overlay = overlay.resize((target_w, target_h), Image.LANCZOS)

            # Opacity (scales the existing alpha channel, preserving transparency shape).
            opacity = self._num(opacity, 1.0)
            if not math.isfinite(opacity):
                opacity = 1.0
            opacity = max(0.0, min(1.0, opacity))
            if opacity < 1.0:
                faded = overlay.split()[3].point(lambda a: int(a * opacity))
                overlay.putalpha(faded)

            margin_value = self._num(margin, max(4, int(base.width * 0.03)))
            if not math.isfinite(margin_value):
                margin_value = max(4, int(base.width * 0.03))
            m = int(margin_value)
            x, y = self._position(position, base.size, overlay.size, m)
            base.alpha_composite(overlay, (x, y))

            out = str(resolve_output_dir(
                output_dir, os.path.splitext(os.path.basename(base_abs))[0] + "_composited.png"))
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            if os.path.splitext(out)[1].lower() in (".jpg", ".jpeg"):
                base.convert("RGB").save(out, quality=95)
            else:
                base.save(out)
        except _CompositeLimitError as e:
            return f"Error: refusing unsafe image composite: {e}"
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
            return f"Error: refusing unsafe image composite: {e}"
        except Exception as e:
            return self.format_error_message(e, "compositing the images")

        return (f"Composited '{os.path.basename(ov_abs)}' onto '{os.path.basename(base_abs)}' at "
                f"{position} ({overlay.width}x{overlay.height}px, opacity {opacity:.2f}). Saved to: {out}")
