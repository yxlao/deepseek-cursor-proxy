"""Generate a clean SVG logo from the PNG draft using vtracer.

Pipeline:
1. Run vtracer to trace the PNG into SVG paths
2. Map all traced colors to the three target colors (#4D6BFE, #6D86FF, #FFFFFF)
3. Apply coordinate transforms (remove translate) for cleaner SVG
4. Replace the eye path with a <circle> element
5. Grid-search optimal eye center and radius
6. Output a clean, well-structured SVG
"""

import re
from pathlib import Path

import cairosvg
import numpy as np
import vtracer
from PIL import Image

ASSETS_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_PNG_PATH = ASSETS_DIR / "logo_draft.png"
SVG_PATH = ASSETS_DIR / "logo.svg"
RENDERED_PNG_PATH = ASSETS_DIR / "logo.png"

DARK_BLUE = "#4D6BFE"
LIGHT_BLUE = "#6D86FF"
WHITE = "#FFFFFF"

COLOR_MAP = {
    "#4D6BFD": DARK_BLUE,
    "#4E6CFB": DARK_BLUE,
    "#4F6DFC": DARK_BLUE,
    "#4D6BFC": DARK_BLUE,
    "#4D6CFE": DARK_BLUE,
    "#4E6DFE": DARK_BLUE,
    "#4D6AFE": DARK_BLUE,
    "#4C6BFE": DARK_BLUE,
    "#6C85FC": LIGHT_BLUE,
    "#6D86FC": LIGHT_BLUE,
    "#6D85FE": LIGHT_BLUE,
    "#6C86FE": LIGHT_BLUE,
    "#6E87FF": LIGHT_BLUE,
    "#6D86FD": LIGHT_BLUE,
    "#6C86FF": LIGHT_BLUE,
    "#6E86FF": LIGHT_BLUE,
    "#FEFEFE": WHITE,
    "#FDFDFD": WHITE,
    "#FCFCFC": WHITE,
}

VTRACER_PARAMS = {
    "colormode": "color",
    "hierarchical": "stacked",
    "mode": "spline",
    "filter_speckle": 4,
    "color_precision": 6,
    "layer_difference": 16,
    "corner_threshold": 60,
    "length_threshold": 4.0,
    "max_iterations": 10,
    "splice_threshold": 45,
    "path_precision": 3,
}

ELEMENT_LABELS = {
    0: "background",
    1: "whale body",
    2: "whale belly",
    3: "smile / lower belly accent",
    4: "cursor arrow - left half",
    5: "cursor arrow - right half",
    6: "eye",
}


def transform_path(d_str: str, tx: float, ty: float) -> str:
    """Apply translate(tx, ty) to absolute path coordinates and round."""
    tokens = re.findall(r"([MCLZmclz])|(-?\d+\.?\d*)", d_str)
    result = []
    cmd = "M"
    idx = 0
    for cmd_tok, num_tok in tokens:
        if cmd_tok:
            cmd = cmd_tok
            idx = 0
            result.append(cmd_tok)
        elif num_tok:
            val = float(num_tok)
            if cmd in "MLCSQT":
                val += tx if idx % 2 == 0 else ty
            elif cmd == "H":
                val += tx
            elif cmd == "V":
                val += ty
            if abs(val - round(val)) < 0.15:
                result.append(str(int(round(val))))
            else:
                result.append(f"{val:.1f}")
            idx += 1
    return " ".join(result)


def compute_mse(svg_path: Path, orig_arr: np.ndarray, w: int, h: int) -> float:
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(RENDERED_PNG_PATH),
        output_width=w,
        output_height=h,
    )
    rend = np.array(Image.open(RENDERED_PNG_PATH).convert("RGB"))
    return float(np.mean((orig_arr.astype(np.float64) - rend.astype(np.float64)) ** 2))


def main() -> None:
    original = Image.open(ORIGINAL_PNG_PATH).convert("RGB")
    w, h = original.size
    orig_arr = np.array(original)
    print(f"Original: {w}x{h}")

    # Step 1: Run vtracer
    raw_path = ASSETS_DIR / "logo_vtracer_raw.svg"
    vtracer.convert_image_to_svg_py(
        image_path=str(ORIGINAL_PNG_PATH),
        out_path=str(raw_path),
        **VTRACER_PARAMS,
    )
    svg_text = raw_path.read_text()

    # Step 2: Map colors
    for old, new in COLOR_MAP.items():
        svg_text = svg_text.replace(old, new)

    # Step 3: Parse and transform paths
    paths = []
    for m in re.finditer(
        r'<path\s+d="([^"]+)"\s+fill="([^"]+)"\s+transform="translate\(([^)]+)\)"',
        svg_text,
    ):
        d, fill, tr = m.groups()
        tx, ty = (float(x) for x in tr.split(","))
        paths.append({"d": d, "fill": fill, "tx": tx, "ty": ty})

    print(f"Traced {len(paths)} paths")

    # Step 4: Identify eye (small white shape in body area)
    eye_idx = None
    for i, p in enumerate(paths):
        if p["fill"] == WHITE and p["tx"] > 150 and p["ty"] > 200:
            eye_idx = i
            break

    # Step 5: Grid search for optimal eye circle
    best_mse = 999.0
    best_cx, best_cy, best_r = 191, 260, 14.0

    # Build SVG template with placeholder eye
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        f'  <rect width="{w}" height="{h}" fill="white"/>',
    ]
    for i, p in enumerate(paths):
        if p["fill"] == WHITE and p["tx"] == 0:
            continue
        if i == eye_idx:
            parts.append("  EYE_PLACEHOLDER")
            continue
        td = transform_path(p["d"], p["tx"], p["ty"])
        label = ELEMENT_LABELS.get(i, "")
        comment = f"  <!-- {label} -->\n" if label else ""
        parts.append(f"{comment}  <path d=\"{td}\" fill=\"{p['fill']}\"/>")
    parts.append("</svg>")
    template = "\n".join(parts)

    print("Searching for optimal eye parameters...")
    for cx in range(189, 195):
        for cy in range(258, 264):
            for r_10 in range(120, 160, 5):
                r = r_10 / 10.0
                test_svg = template.replace(
                    "  EYE_PLACEHOLDER",
                    f'  <!-- eye -->\n  <circle cx="{cx}" cy="{cy}" r="{r}" fill="white"/>',
                )
                tmp = ASSETS_DIR / "logo_tmp.svg"
                tmp.write_text(test_svg)
                mse = compute_mse(tmp, orig_arr, w, h)
                if mse < best_mse:
                    best_mse = mse
                    best_cx, best_cy, best_r = cx, cy, r

    print(f"Best eye: cx={best_cx}, cy={best_cy}, r={best_r}")

    # Step 6: Write final SVG
    final_svg = template.replace(
        "  EYE_PLACEHOLDER",
        f"  <!-- eye -->\n"
        f'  <circle cx="{best_cx}" cy="{best_cy}" r="{best_r}" fill="white"/>',
    )
    SVG_PATH.write_text(final_svg)

    # Compute final MSE
    mse = compute_mse(SVG_PATH, orig_arr, w, h)
    similarity = (1 - mse / 255.0**2) * 100
    print(f"\nFinal MSE: {mse:.2f}")
    print(f"Similarity: {similarity:.2f}%")
    print(f"SVG size: {len(final_svg)} bytes")

    # Cleanup
    raw_path.unlink(missing_ok=True)
    (ASSETS_DIR / "logo_tmp.svg").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
