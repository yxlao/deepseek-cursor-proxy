"""Compare rendered SVG logo against the original PNG logo using MSE."""

from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_PNG_PATH = ASSETS_DIR / "logo_draft.png"
SVG_PATH = ASSETS_DIR / "logo.svg"
RENDERED_PNG_PATH = ASSETS_DIR / "logo.png"
DIFF_PNG_PATH = ASSETS_DIR / "logo_diff.png"


def render_svg_to_png(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        output_height=height,
    )


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def main() -> None:
    original = Image.open(ORIGINAL_PNG_PATH).convert("RGB")
    width, height = original.size
    print(f"Original PNG size: {width}x{height}")

    render_svg_to_png(SVG_PATH, RENDERED_PNG_PATH, width, height)
    rendered = Image.open(RENDERED_PNG_PATH).convert("RGB")
    print(f"Rendered PNG size: {rendered.size[0]}x{rendered.size[1]}")

    orig_arr = np.array(original)
    rend_arr = np.array(rendered)

    mse_all = compute_mse(orig_arr, rend_arr)
    print(f"\nOverall MSE: {mse_all:.2f}")

    for ch, name in enumerate(["Red", "Green", "Blue"]):
        ch_mse = compute_mse(orig_arr[:, :, ch], rend_arr[:, :, ch])
        print(f"  {name} channel MSE: {ch_mse:.2f}")

    max_possible_mse = 255.0**2
    similarity = (1 - mse_all / max_possible_mse) * 100
    print(f"\nSimilarity: {similarity:.2f}%")

    diff = np.abs(orig_arr.astype(np.float64) - rend_arr.astype(np.float64))
    diff_mean = diff.mean(axis=2)
    high_diff_mask = diff_mean > 50
    high_diff_count = high_diff_mask.sum()
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
            f"High-diff region bounding box: x=[{x_min}-{x_max}], y=[{y_min}-{y_max}]"
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

    diff_scaled = np.clip(diff * 3, 0, 255).astype(np.uint8)
    Image.fromarray(diff_scaled).save(DIFF_PNG_PATH)
    print(f"\nDiff image saved to {DIFF_PNG_PATH.name}")


if __name__ == "__main__":
    main()
