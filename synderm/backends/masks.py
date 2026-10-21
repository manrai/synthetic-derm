"""Masks for the paper's inpaint and in-then-outpaint methods (white = regenerate)."""
from PIL import Image, ImageDraw


def center_mask(size: tuple[int, int], fraction: float = 0.5) -> Image.Image:
    """White square in the centre covering ``fraction`` of each side (the paper used one half)."""
    w, h = size
    m = Image.new("L", (w, h), 0)
    bw, bh = int(w * fraction), int(h * fraction)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    ImageDraw.Draw(m).rectangle([x0, y0, x0 + bw, y0 + bh], fill=255)
    return m


def border_mask(size: tuple[int, int], fraction: float = 0.5) -> Image.Image:
    """Exact complement of :func:`center_mask`: everything outside the centre square."""
    m = center_mask(size, fraction)
    return Image.eval(m, lambda v: 255 - v)
