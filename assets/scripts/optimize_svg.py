"""Generate a clean SVG logo from the PNG draft.

Pipeline:
1. Run vtracer to trace whale body/belly/smile paths
2. Clean up whale paths (apply translate, merge tiny segments)
3. Generate cursor arrow mathematically (perfect left-right symmetry, rounded corners)
4. Add circle eye (grid-search optimal position)
5. Add white margins around the logo
6. Output clean SVG and rendered PNG
7. Clean up all intermediate files
"""

import math
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

MARGIN = 20

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


# ---------------------------------------------------------------------------
# Path cleaning: apply translate, merge tiny segments, round coordinates
# ---------------------------------------------------------------------------


def _point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
    """Distance from point (px,py) to line segment (lx1,ly1)-(lx2,ly2)."""
    dx, dy = lx2 - lx1, ly2 - ly1
    ll = dx * dx + dy * dy
    if ll < 1e-12:
        return math.hypot(px - lx1, py - ly1)
    t = max(0, min(1, ((px - lx1) * dx + (py - ly1) * dy) / ll))
    return math.hypot(px - (lx1 + t * dx), py - (ly1 + t * dy))


def _is_cubic_near_linear(cursor, cp1, cp2, end, threshold=2.0):
    """Check if a cubic Bézier is well-approximated by a straight line."""
    d1 = _point_to_line_dist(cp1[0], cp1[1], cursor[0], cursor[1], end[0], end[1])
    d2 = _point_to_line_dist(cp2[0], cp2[1], cursor[0], cursor[1], end[0], end[1])
    return max(d1, d2) < threshold


def transform_and_clean_path(
    d_str: str,
    tx: float,
    ty: float,
    linearity_threshold: float = 2.0,
    collinearity_threshold: float = 1.5,
    min_segment_len: float = 1.0,
) -> str:
    """Apply translate, simplify near-linear cubics to lines, merge collinear lines."""
    tokens = re.findall(r"([MCLZmclz])|(-?\d+\.?\d*)", d_str)
    raw_cmds = []
    cmd = "M"
    nums = []
    for cmd_tok, num_tok in tokens:
        if cmd_tok:
            if nums:
                raw_cmds.append((cmd, nums))
            cmd = cmd_tok
            nums = []
        elif num_tok:
            nums.append(float(num_tok))
    if nums:
        raw_cmds.append((cmd, nums))

    # Phase 1: Apply translate, replace near-linear cubics with lines
    phase1 = []
    cursor = (0.0, 0.0)
    for cmd, nums in raw_cmds:
        if cmd == "M":
            x, y = nums[0] + tx, nums[1] + ty
            cursor = (x, y)
            phase1.append(("M", (x, y)))
        elif cmd == "L":
            for i in range(0, len(nums), 2):
                end = (nums[i] + tx, nums[i + 1] + ty)
                if (
                    math.hypot(end[0] - cursor[0], end[1] - cursor[1])
                    >= min_segment_len
                ):
                    phase1.append(("L", end))
                    cursor = end
        elif cmd == "C":
            for i in range(0, len(nums), 6):
                if i + 5 < len(nums):
                    cp1 = (nums[i] + tx, nums[i + 1] + ty)
                    cp2 = (nums[i + 2] + tx, nums[i + 3] + ty)
                    end = (nums[i + 4] + tx, nums[i + 5] + ty)
                    seg_len = math.hypot(end[0] - cursor[0], end[1] - cursor[1])
                    if seg_len < min_segment_len:
                        continue
                    if _is_cubic_near_linear(
                        cursor, cp1, cp2, end, linearity_threshold
                    ):
                        phase1.append(("L", end))
                    else:
                        phase1.append(("C", (cp1, cp2, end)))
                    cursor = end
        elif cmd == "Z":
            phase1.append(("Z", None))

    # Phase 2: Merge consecutive collinear L segments
    phase2 = []
    i = 0
    while i < len(phase1):
        cmd, data = phase1[i]
        if cmd != "L":
            phase2.append((cmd, data))
            i += 1
            continue
        # Collect consecutive L segments
        run_start = i - 1
        start_pt = phase2[-1][1] if phase2 and phase2[-1][0] in ("M", "L") else None
        if start_pt is None:
            phase2.append((cmd, data))
            i += 1
            continue
        if isinstance(start_pt, tuple) and len(start_pt) == 2:
            origin = start_pt
        else:
            phase2.append((cmd, data))
            i += 1
            continue

        run = [data]
        j = i + 1
        while j < len(phase1) and phase1[j][0] == "L":
            run.append(phase1[j][1])
            j += 1

        # Greedy merge: keep extending line as long as all intermediate points are close
        merged = []
        seg_start = origin
        k = 0
        while k < len(run):
            best_end = k
            for e in range(k + 1, len(run)):
                all_close = True
                for m_idx in range(k, e):
                    d = _point_to_line_dist(
                        run[m_idx][0],
                        run[m_idx][1],
                        seg_start[0],
                        seg_start[1],
                        run[e][0],
                        run[e][1],
                    )
                    if d > collinearity_threshold:
                        all_close = False
                        break
                if all_close:
                    best_end = e
                else:
                    break
            merged.append(("L", run[best_end]))
            seg_start = run[best_end]
            k = best_end + 1

        phase2.extend(merged)
        i = j

    # Format output
    parts = []
    for cmd, data in phase2:
        if cmd == "M":
            parts.append(f"M{_fmt_coord(data[0])} {_fmt_coord(data[1])}")
        elif cmd == "L":
            parts.append(f"L{_fmt_coord(data[0])} {_fmt_coord(data[1])}")
        elif cmd == "C":
            cp1, cp2, end = data
            parts.append(
                f"C{_fmt_coord(cp1[0])} {_fmt_coord(cp1[1])} "
                f"{_fmt_coord(cp2[0])} {_fmt_coord(cp2[1])} "
                f"{_fmt_coord(end[0])} {_fmt_coord(end[1])}"
            )
        elif cmd == "Z":
            parts.append("Z")
    return " ".join(parts)


def _fmt_coord(v):
    if abs(v - round(v)) < 0.05:
        return str(int(round(v)))
    return f"{v:.1f}"


# ---------------------------------------------------------------------------
# Cursor: mathematically symmetric with rounded corners
# ---------------------------------------------------------------------------


def generate_cursor_paths():
    """Generate perfectly symmetric cursor arrow with rounded corners.

    Returns (left_half_d, right_half_d).
    Left half = dark blue, Right half = light blue.

    Geometry derived from linear regression on the original PNG edges:
      Left diagonal:  x = -0.4783*y + 168.33
      Right diagonal: x =  0.4783*y + 174.19
      Stem left:  x =  0.4011*y + 116.91
      Stem right: x = -0.4011*y + 226.55
    Symmetry center: x = 171.26
    """
    cx = 171.3
    diag_slope = 0.4783
    diag_intercept = 2.93  # half-width at y=0
    stem_slope = 0.4011
    stem_intercept = 54.82  # half-width extrapolated constant

    tip_y = 0
    wing_y = 103
    wing_base_y = 105
    stem_top_y = 107
    stem_bottom_y = 131
    r_wing = 3
    r_inner = 3
    r_stem = 1.5

    # Wing outer corner
    wing_hw = diag_slope * wing_y + diag_intercept  # ~52.2
    left_wing = cx - wing_hw
    right_wing = cx + wing_hw

    # Wing base (slightly inward due to rounding in original)
    base_hw = (
        diag_slope * wing_base_y + diag_intercept
    )  # ~53.2 → clip to match original ~49.5
    base_hw = min(base_hw, wing_hw)

    # Stem top
    stem_top_hw = stem_intercept - stem_slope * stem_top_y  # ~11.9
    left_stem_top = cx - stem_top_hw
    right_stem_top = cx + stem_top_hw

    # Stem bottom
    stem_bot_hw = stem_intercept - stem_slope * stem_bottom_y  # ~2.3
    left_stem_bot = cx - stem_bot_hw
    right_stem_bot = cx + stem_bot_hw

    # Tip half-width at y=0
    tip_hw = diag_intercept

    # Diagonal direction (pointing downward-right from center)
    diag_dy = wing_y - tip_y
    diag_dx = wing_hw - tip_hw
    diag_len = math.hypot(diag_dx, diag_dy)
    du = diag_dx / diag_len
    dv = diag_dy / diag_len

    # Stem direction (pointing downward, narrowing)
    stem_dy = stem_bottom_y - stem_top_y
    stem_dx = stem_top_hw - stem_bot_hw
    stem_len = math.hypot(stem_dx, stem_dy)
    su = stem_dx / stem_len
    sv = stem_dy / stem_len

    def fmt(x, y):
        return f"{x:.1f} {y:.1f}"

    # --- Left wing outer corner (left_wing, wing_y) ---
    lw_bx = left_wing + r_wing * du
    lw_by = wing_y - r_wing * dv
    lw_ax = left_wing + r_wing
    lw_ay = wing_y

    # --- Left wing inner corner (left_stem_top, wing_base_y → stem_top_y) ---
    li_bx = left_stem_top - r_inner
    li_by = wing_base_y
    li_ax = left_stem_top + r_inner * su
    li_ay = stem_top_y + r_inner * sv

    # --- Left stem bottom corner ---
    lb_bx = left_stem_bot - r_stem * su
    lb_by = stem_bottom_y - r_stem * sv
    lb_ax = cx
    lb_ay = stem_bottom_y

    # --- Right wing outer corner ---
    rw_bx = right_wing - r_wing
    rw_by = wing_y
    rw_ax = right_wing - r_wing * du
    rw_ay = wing_y - r_wing * dv

    # --- Right wing inner corner ---
    ri_bx = right_stem_top - r_inner * su
    ri_by = stem_top_y + r_inner * sv
    ri_ax = right_stem_top + r_inner
    ri_ay = wing_base_y

    # --- Right stem bottom corner ---
    rb_bx = cx
    rb_by = stem_bottom_y
    rb_ax = right_stem_bot + r_stem * su
    rb_ay = stem_bottom_y - r_stem * sv

    left_d = (
        f"M{cx} {tip_y} "
        f"L{fmt(lw_bx, lw_by)} "
        f"Q{fmt(left_wing, wing_y)} {fmt(lw_ax, lw_ay)} "
        f"L{fmt(li_bx, li_by)} "
        f"Q{fmt(left_stem_top, wing_base_y)} {fmt(li_ax, li_ay)} "
        f"L{fmt(lb_bx, lb_by)} "
        f"Q{fmt(left_stem_bot, stem_bottom_y)} {fmt(lb_ax, lb_ay)} "
        "Z"
    )

    right_d = (
        f"M{cx} {tip_y} "
        f"L{cx} {stem_bottom_y} "
        f"Q{fmt(right_stem_bot, stem_bottom_y)} {fmt(rb_ax, rb_ay)} "
        f"L{fmt(ri_bx, ri_by)} "
        f"Q{fmt(right_stem_top, wing_base_y)} {fmt(ri_ax, ri_ay)} "
        f"L{fmt(rw_bx, rw_by)} "
        f"Q{fmt(right_wing, wing_y)} {fmt(rw_ax, rw_ay)} "
        "Z"
    )

    return left_d, right_d


# ---------------------------------------------------------------------------
# SVG assembly and optimization
# ---------------------------------------------------------------------------


def render_and_mse(
    svg_text: str, orig_arr: np.ndarray, w: int, h: int, tmp_svg: Path, tmp_png: Path
) -> float:
    tmp_svg.write_text(svg_text)
    cairosvg.svg2png(
        url=str(tmp_svg), write_to=str(tmp_png), output_width=w, output_height=h
    )
    rend = np.array(Image.open(tmp_png).convert("RGB"))
    return float(np.mean((orig_arr.astype(np.float64) - rend.astype(np.float64)) ** 2))


def main() -> None:
    original = Image.open(ORIGINAL_PNG_PATH).convert("RGB")
    orig_w, orig_h = original.size
    orig_arr = np.array(original)
    print(f"Original: {orig_w}x{orig_h}")

    tmp_svg = ASSETS_DIR / "_tmp.svg"
    tmp_png = ASSETS_DIR / "_tmp.png"

    # --- Step 1: Run vtracer ---
    raw_path = ASSETS_DIR / "_vtracer_raw.svg"
    vtracer.convert_image_to_svg_py(
        image_path=str(ORIGINAL_PNG_PATH), out_path=str(raw_path), **VTRACER_PARAMS
    )
    svg_text = raw_path.read_text()
    for old, new in COLOR_MAP.items():
        svg_text = svg_text.replace(old, new)

    paths = []
    for m in re.finditer(
        r'<path\s+d="([^"]+)"\s+fill="([^"]+)"\s+transform="translate\(([^)]+)\)"',
        svg_text,
    ):
        d, fill, tr = m.groups()
        tx, ty = (float(x) for x in tr.split(","))
        paths.append({"d": d, "fill": fill, "tx": tx, "ty": ty})
    print(f"Traced {len(paths)} paths")

    # paths: 0=bg, 1=body(dark), 2=belly(light), 3=smile(dark), 4=cursor_dark, 5=cursor_light, 6=eye

    # --- Step 2: Clean whale paths ---
    body_d = transform_and_clean_path(
        paths[1]["d"], paths[1]["tx"], paths[1]["ty"], min_segment_len=1.5
    )
    belly_d = transform_and_clean_path(
        paths[2]["d"], paths[2]["tx"], paths[2]["ty"], min_segment_len=1.5
    )
    smile_d = transform_and_clean_path(
        paths[3]["d"], paths[3]["tx"], paths[3]["ty"], min_segment_len=1.0
    )

    orig_body_segs = paths[1]["d"].count("C") + paths[1]["d"].count("L")
    orig_belly_segs = paths[2]["d"].count("C") + paths[2]["d"].count("L")
    orig_smile_segs = paths[3]["d"].count("C") + paths[3]["d"].count("L")
    new_body_segs = body_d.count("C") + body_d.count("L")
    new_belly_segs = belly_d.count("C") + belly_d.count("L")
    new_smile_segs = smile_d.count("C") + smile_d.count("L")
    print(f"Body: {orig_body_segs} -> {new_body_segs} segments")
    print(f"Belly: {orig_belly_segs} -> {new_belly_segs} segments")
    print(f"Smile: {orig_smile_segs} -> {new_smile_segs} segments")

    # --- Step 3: Generate cursor ---
    cursor_left_d, cursor_right_d = generate_cursor_paths()
    print("Generated symmetric cursor")

    # --- Step 4: Grid-search eye ---
    def build_svg_no_margin(cx_eye, cy_eye, r_eye):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {orig_w} {orig_h}" '
            f'width="{orig_w}" height="{orig_h}">\n'
            f'  <rect width="{orig_w}" height="{orig_h}" fill="white"/>\n\n'
            f'  <!-- Whale body -->\n  <path d="{body_d}" fill="{DARK_BLUE}"/>\n\n'
            f'  <!-- Whale belly -->\n  <path d="{belly_d}" fill="{LIGHT_BLUE}"/>\n\n'
            f'  <!-- Smile -->\n  <path d="{smile_d}" fill="{DARK_BLUE}"/>\n\n'
            f'  <!-- Eye -->\n  <circle cx="{cx_eye}" cy="{cy_eye}" r="{r_eye}" fill="white"/>\n\n'
            f'  <!-- Cursor left half (dark blue) -->\n  <path d="{cursor_left_d}" fill="{DARK_BLUE}"/>\n\n'
            f'  <!-- Cursor right half (light blue) -->\n  <path d="{cursor_right_d}" fill="{LIGHT_BLUE}"/>\n'
            f"</svg>"
        )

    best_mse, best_cx, best_cy, best_r = 999.0, 191, 260, 14.0
    print("Searching for optimal eye parameters...")
    for cx_eye in range(188, 196):
        for cy_eye in range(257, 264):
            for r_10 in range(115, 160, 5):
                r_eye = r_10 / 10.0
                test = build_svg_no_margin(cx_eye, cy_eye, r_eye)
                mse = render_and_mse(test, orig_arr, orig_w, orig_h, tmp_svg, tmp_png)
                if mse < best_mse:
                    best_mse = mse
                    best_cx, best_cy, best_r = cx_eye, cy_eye, r_eye
    print(f"Best eye: cx={best_cx}, cy={best_cy}, r={best_r}, MSE={best_mse:.2f}")

    # --- Step 5: Write final SVG with margins ---
    m = MARGIN
    final_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{-m} {-m} {orig_w + 2*m} {orig_h + 2*m}" '
        f'width="{orig_w + 2*m}" height="{orig_h + 2*m}">\n'
        f'  <rect x="{-m}" y="{-m}" width="{orig_w + 2*m}" height="{orig_h + 2*m}" fill="white"/>\n\n'
        f'  <!-- Whale body -->\n  <path d="{body_d}" fill="{DARK_BLUE}"/>\n\n'
        f'  <!-- Whale belly -->\n  <path d="{belly_d}" fill="{LIGHT_BLUE}"/>\n\n'
        f'  <!-- Smile -->\n  <path d="{smile_d}" fill="{DARK_BLUE}"/>\n\n'
        f'  <!-- Eye -->\n  <circle cx="{best_cx}" cy="{best_cy}" r="{best_r}" fill="white"/>\n\n'
        f'  <!-- Cursor left half (dark blue) -->\n  <path d="{cursor_left_d}" fill="{DARK_BLUE}"/>\n\n'
        f'  <!-- Cursor right half (light blue) -->\n  <path d="{cursor_right_d}" fill="{LIGHT_BLUE}"/>\n'
        f"</svg>"
    )
    SVG_PATH.write_text(final_svg)

    # Render logo.png from the with-margin SVG
    cairosvg.svg2png(url=str(SVG_PATH), write_to=str(RENDERED_PNG_PATH))

    # Report final MSE (no-margin comparison)
    similarity = (1 - best_mse / 255.0**2) * 100
    print(f"\nFinal MSE (vs original): {best_mse:.2f}")
    print(f"Similarity: {similarity:.2f}%")
    print(f"SVG size: {len(final_svg)} bytes")

    # --- Step 6: Cleanup ---
    for p in [raw_path, tmp_svg, tmp_png]:
        p.unlink(missing_ok=True)
    print("Cleaned up intermediate files")


if __name__ == "__main__":
    main()
