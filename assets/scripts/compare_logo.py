"""Compare rendered SVG logo against the original PNG logo using MSE.

Also generates:
- logo_diff.png: pixel-level difference image (3x amplified)
- logo_grad_diff.png: gradient (edge) difference image
"""

from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

ASSETS_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_PNG_PATH = ASSETS_DIR / "logo_draft.png"
SVG_PATH = ASSETS_DIR / "logo.svg"
RENDERED_PNG_PATH = ASSETS_DIR / "logo.png"
DIFF_PNG_PATH = ASSETS_DIR / "logo_diff.png"
GRAD_DIFF_PNG_PATH = ASSETS_DIR / "logo_grad_diff.png"


def render_svg_to_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height,
    )


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude on a grayscale image."""
    gx = sobel(gray.astype(np.float64), axis=1)
    gy = sobel(gray.astype(np.float64), axis=0)
    return np.sqrt(gx**2 + gy**2)


def main() -> None:
    original = Image.open(ORIGINAL_PNG_PATH).convert("RGB")
    width, height = original.size
    print(f"Original PNG size: {width}x{height}")

    # Render the SVG without margins for pixel-accurate comparison
    svg_text = SVG_PATH.read_text()

    import re as _re

    vb_match = _re.search(r'viewBox="([^"]+)"', svg_text)
    if vb_match:
        parts = vb_match.group(1).split()
        vb_x, vb_y = float(parts[0]), float(parts[1])
        has_margin = vb_x < 0 or vb_y < 0
    else:
        has_margin = False

    if has_margin:
        inner_w = float(parts[2]) + 2 * vb_x  # subtract margins from each side
        inner_h = float(parts[3]) + 2 * vb_y
        # Build a no-margin SVG by only changing the <svg> tag attributes
        no_margin = _re.sub(
            r"<svg\s[^>]+>",
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {inner_w:.0f} {inner_h:.0f}" '
            f'width="{inner_w:.0f}" height="{inner_h:.0f}">',
            svg_text,
            count=1,
        )
        tmp_svg = ASSETS_DIR / "_compare_tmp.svg"
        tmp_svg.write_text(no_margin)
        render_svg_to_png(tmp_svg, RENDERED_PNG_PATH, width, height)
        tmp_svg.unlink(missing_ok=True)
    else:
        render_svg_to_png(SVG_PATH, RENDERED_PNG_PATH, width, height)

    rendered = Image.open(RENDERED_PNG_PATH).convert("RGB")
    print(f"Rendered PNG size: {rendered.size[0]}x{rendered.size[1]}")

    orig_arr = np.array(original)
    rend_arr = np.array(rendered)

    # --- Photometric MSE ---
    mse_all = compute_mse(orig_arr, rend_arr)
    print(f"\nOverall MSE: {mse_all:.2f}")
    for ch, name in enumerate(["Red", "Green", "Blue"]):
        ch_mse = compute_mse(orig_arr[:, :, ch], rend_arr[:, :, ch])
        print(f"  {name} channel MSE: {ch_mse:.2f}")

    similarity = (1 - mse_all / 255.0**2) * 100
    print(f"\nSimilarity: {similarity:.2f}%")

    # --- Pixel diff analysis ---
    diff = np.abs(orig_arr.astype(np.float64) - rend_arr.astype(np.float64))
    diff_mean = diff.mean(axis=2)
    high_diff_mask = diff_mean > 50
    high_diff_count = int(high_diff_mask.sum())
    total_pixels = width * height
    print(
        f"Pixels with mean diff > 50: {high_diff_count} ({high_diff_count/total_pixels*100:.1f}%)"
    )

    pct_perfect = (diff.max(axis=2) == 0).sum() / total_pixels * 100
    print(f"Pixels with zero diff: {pct_perfect:.1f}%")

    if high_diff_count > 0:
        coords = np.argwhere(high_diff_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        print(
            f"\nHigh-diff region bounding box: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}]"
        )
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2
        regions = {
            "top-left": high_diff_mask[:y_center, :x_center].sum(),
            "top-right": high_diff_mask[:y_center, x_center:].sum(),
            "bottom-left": high_diff_mask[y_center:, :x_center].sum(),
            "bottom-right": high_diff_mask[y_center:, x_center:].sum(),
        }
        for region_name, count in sorted(regions.items(), key=lambda x: -x[1]):
            print(f"  {region_name}: {count} pixels")

    # --- Diff image ---
    diff_scaled = np.clip(diff * 3, 0, 255).astype(np.uint8)
    Image.fromarray(diff_scaled).save(DIFF_PNG_PATH)
    print(f"\nDiff image saved to {DIFF_PNG_PATH.name}")

    # --- Gradient (edge) diff ---
    orig_gray = np.mean(orig_arr.astype(np.float64), axis=2)
    rend_gray = np.mean(rend_arr.astype(np.float64), axis=2)
    grad_orig = gradient_magnitude(orig_gray)
    grad_rend = gradient_magnitude(rend_gray)
    grad_diff = np.abs(grad_orig - grad_rend)
    grad_mse = float(np.mean(grad_diff**2))
    print(f"Gradient diff MSE: {grad_mse:.2f}")

    grad_vis = np.clip(grad_diff * 5, 0, 255).astype(np.uint8)
    grad_vis_rgb = np.stack([grad_vis, grad_vis, grad_vis], axis=2)
    Image.fromarray(grad_vis_rgb).save(GRAD_DIFF_PNG_PATH)
    print(f"Gradient diff image saved to {GRAD_DIFF_PNG_PATH.name}")

    # Re-render logo.png from the final SVG (with margins if present)
    cairosvg.svg2png(url=str(SVG_PATH), write_to=str(RENDERED_PNG_PATH))
    print("\nFinal logo.png rendered from SVG")


if __name__ == "__main__":
    main()
