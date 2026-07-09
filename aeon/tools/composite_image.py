import os
from PIL import Image
from .base import BaseTool
from ..core.paths import resolve_output_dir


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
            base = Image.open(base_abs).convert("RGBA")
            overlay = Image.open(ov_abs).convert("RGBA")

            # Resize overlay: scale <=1 is a fraction of base width; >1 is a pixel width.
            scale = self._num(scale, 0.2)
            if scale <= 0:
                scale = 0.2
            target_w = max(1, int(base.width * scale)) if scale <= 1.0 else min(int(scale), base.width)
            target_h = max(1, round(overlay.height * (target_w / overlay.width)))
            overlay = overlay.resize((target_w, target_h), Image.LANCZOS)

            # Opacity (scales the existing alpha channel, preserving transparency shape).
            opacity = max(0.0, min(1.0, self._num(opacity, 1.0)))
            if opacity < 1.0:
                faded = overlay.split()[3].point(lambda a: int(a * opacity))
                overlay.putalpha(faded)

            m = int(self._num(margin, max(4, int(base.width * 0.03))))
            x, y = self._position(position, base.size, overlay.size, m)
            base.alpha_composite(overlay, (x, y))

            out = str(resolve_output_dir(
                output_dir, os.path.splitext(os.path.basename(base_abs))[0] + "_composited.png"))
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            if os.path.splitext(out)[1].lower() in (".jpg", ".jpeg"):
                base.convert("RGB").save(out, quality=95)
            else:
                base.save(out)
        except Exception as e:
            return self.format_error_message(e, "compositing the images")

        return (f"Composited '{os.path.basename(ov_abs)}' onto '{os.path.basename(base_abs)}' at "
                f"{position} ({overlay.width}x{overlay.height}px, opacity {opacity:.2f}). Saved to: {out}")
