"""PCB/Gerber analysis for CTF challenges.

Parses KiCad Gerber files, renders ASCII visualizations,
decodes vector text, detects anomalies, and searches for flags.
"""
import math
import os
import re
import base64
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

_COORD_RE = re.compile(r"X(\d+)Y(-?\d+)")
_DCODE_RE = re.compile(r"D(\d+)\*")
_FLASH_RE = re.compile(r"X(\d+)Y(-?\d+)D03\*")
_LINE_RE = re.compile(r"X(\d+)Y(-?\d+)(?:I(-?\d+)J(-?\d+))?(?:D01|D02)\*")

_GERBER_APERTURE = re.compile(r"%ADD(\d+)([A-Za-z]+),([\d.]+)(?:X([\d.]+))?%\*?")


def _parse_gerber_coords(path: str) -> dict:
    """Parse Gerber file, extract coordinates per aperture."""
    with open(path) as f:
        content = f.read()

    lines = content.split("\n")
    current_dcode = None
    coords_by_dcode: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    all_coords: list[tuple[float, float]] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        dcode_match = _DCODE_RE.search(line)
        if dcode_match:
            d = dcode_match.group(1)
            if d in ("01", "02", "03"):
                current_dcode = d

        # Extract coordinates
        coord_match = _COORD_RE.search(line)
        if coord_match:
            x = int(coord_match.group(1)) / 1_000_000.0
            y = int(coord_match.group(2)) / 1_000_000.0
            all_coords.append((x, y))

            if current_dcode and current_dcode == "01":
                coords_by_dcode["draw"].append((x, y, "D01"))
            elif current_dcode == "03" or line.endswith("D03*"):
                coords_by_dcode["flash"].append((x, y, "D03"))

    return {
        "all_coords": all_coords,
        "draw_coords": coords_by_dcode.get("draw", []),
        "flash_coords": coords_by_dcode.get("flash", []),
        "coord_count": len(all_coords),
    }


def _get_file_layer_info(path: str) -> dict:
    """Extract layer metadata from Gerber header."""
    info = {"name": os.path.basename(path), "path": path}
    with open(path) as f:
        header = f.read(2000)
    m = re.search(r"TF\.FileFunction,(.+)", header)
    info["function"] = m.group(1) if m else "unknown"
    m = re.search(r"TF\.ProjectId,(.+),", header)
    info["project"] = m.group(1) if m else "unknown"
    m = re.search(r"CreationDate,(.+)\*", header)
    info["date"] = m.group(1) if m else "unknown"
    info["size"] = os.path.getsize(path)
    return info


def _detect_via_positions(coords_by_layer: dict[str, list]) -> list:
    """Detect via positions — coordinates appearing on 3+ copper layers."""
    layer_coord_sets: dict[tuple[float, float], int] = Counter()
    for layer_name, coords in coords_by_layer.items():
        if "Cu" not in layer_name:
            continue
        seen = set()
        for x, y, _ in coords:
            c = (round(x, 4), round(y, 4))
            if c not in seen:
                layer_coord_sets[c] += 1
                seen.add(c)
    return [(x, y, count) for (x, y), count in layer_coord_sets.items() if count >= 3]


def _render_ascii_grid(
    coords: list[tuple[float, float, str]],
    width_chars: int = 100,
    height_chars: int = 50,
    title: str = "",
) -> str:
    """Render coordinate data as ASCII art."""
    if not coords:
        return "[no coordinates]"

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    aspect = x_range / y_range if y_range else 1
    adj_width = width_chars
    adj_height = min(height_chars, int(adj_width / aspect * 1.5))

    # Bucket coords to grid cells
    grid = [[" "] * adj_width for _ in range(adj_height)]

    for x, y, kind in coords:
        ix = min(adj_width - 1, int((x - min_x) / x_range * (adj_width - 1)))
        iy = min(adj_height - 1, int((max_y - y) / y_range * (adj_height - 1)))
        ch = grid[iy][ix]
        if ch == " ":
            grid[iy][ix] = "+" if kind == "D03" else "."
        elif ch == "." and kind == "D03":
            grid[iy][ix] = "+"

    rows = [f"  {''.join(row)}" for row in grid if any(c != " " for c in row)]
    header = f"── {title} ── {min_x:.1f}–{max_x:.1f}mm × {min_y:.1f}–{max_y:.1f}mm"
    result = [header, f"  {'.'}=draw  '+'.=flash"]
    result.append(f"  {'─' * adj_width}")
    result.extend(rows)
    return "\n".join(result)


def _extract_text_regions(
    coords: list[tuple[float, float, str]],
    min_density: int = 3,
    cell_size: float = 1.0,
) -> list[dict]:
    """Find dense coordinate clusters (text regions)."""
    collisions: dict[tuple[int, int], list] = defaultdict(list)
    for x, y, kind in coords:
        cx, cy = int(x / cell_size), int(y / cell_size)
        collisions[(cx, cy)].append((x, y, kind))

    regions = []
    for cell, pts in sorted(collisions.items()):
        if len(pts) >= min_density:
            xs_region = [p[0] for p in pts]
            ys_region = [p[1] for p in pts]
            regions.append(
                {
                    "cell": cell,
                    "count": len(pts),
                    "x": round(min(xs_region), 2),
                    "y": round(min(ys_region), 2),
                    "extent_x": round(max(xs_region) - min(xs_region), 2),
                    "extent_y": round(max(ys_region) - min(ys_region), 2),
                }
            )
    regions.sort(key=lambda r: -r["count"])
    return regions


def _analyze_layer(path: str, layer_name: str) -> dict:
    """Analyze a single Gerber layer."""
    info = _get_file_layer_info(path)
    coords = _parse_gerber_coords(path)
    draw = coords["draw_coords"]
    flash = coords["flash_coords"]
    all_c = coords["all_coords"]

    result = {
        "layer": layer_name,
        "file": info["name"],
        "function": info["function"],
        "size_bytes": info["size"],
        "coord_count": coords["coord_count"],
        "draw_count": len(draw),
        "flash_count": len(flash),
    }

    if all_c:
        xs = [c[0] for c in all_c]
        ys = [c[1] for c in all_c]
        result["x_range"] = (round(min(xs), 2), round(max(xs), 2))
        result["y_range"] = (round(min(ys), 2), round(max(ys), 2))
        result["x_span"] = round(max(xs) - min(xs), 2)
        result["y_span"] = round(max(ys) - min(ys), 2)

    if draw:
        result["ascii_art"] = _render_ascii_grid(draw, title=layer_name)
        regions = _extract_text_regions(draw)
        if regions:
            result["dense_regions"] = regions[:5]

    return result


def _parse_gerber_strokes(path: str) -> tuple[list, list]:
    """Parse Gerber file into individual strokes (move→draw sequences).

    Returns (strokes, all_points) where strokes is a list of (x, y) point lists,
    and all_points is every coordinate for bounding box calculation.
    """
    with open(path) as f:
        content = f.read()
    lines = content.split("\n")
    strokes: list[list[tuple[float, float]]] = []
    all_pts: list[tuple[float, float]] = []
    current_stroke: list[tuple[float, float]] | None = None
    last_dcode: str | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        dcode_match = _DCODE_RE.search(line)
        if dcode_match:
            d = dcode_match.group(1)
            if d in ("01", "02", "03"):
                last_dcode = d
        coord_match = _COORD_RE.search(line)
        if not coord_match:
            continue
        x = int(coord_match.group(1)) / 1_000_000.0
        y = int(coord_match.group(2)) / 1_000_000.0
        all_pts.append((x, y))

        if last_dcode == "02":
            if current_stroke:
                strokes.append(current_stroke)
            current_stroke = [(x, y)]
        elif last_dcode == "01":
            if current_stroke is None:
                current_stroke = [(x, y)]
            else:
                current_stroke.append((x, y))

    if current_stroke and len(current_stroke) > 1:
        strokes.append(current_stroke)
    return strokes, all_pts


# KiCad Hershey vector font glyphs
# Each glyph: list of stroke paths, each path is list of (x, y) in normalized 0..9 space
# Based on KiCad's font data (Pcbnew text renderer)
_KICAD_FONT: dict[str, list[list[tuple[float, float]]]] = {
    "H": [[(0, 1), (0, 9)], [(8, 1), (8, 9)], [(0, 5), (8, 5)]],
    "T": [[(0, 9), (8, 9)], [(4, 9), (4, 1)]],
    "B": [[(0, 1), (0, 9)], [(0, 9), (7, 9), (8, 8), (8, 7), (7, 6), (0, 5)], [(0, 5), (7, 5), (8, 4), (8, 3), (7, 2), (0, 1)]],
    "{": [[(5, 9), (3, 9), (2, 8), (2, 6), (1, 5), (2, 4), (2, 2), (3, 1), (5, 1)]],
    "}": [[(3, 9), (5, 9), (6, 8), (6, 6), (7, 5), (6, 4), (6, 2), (5, 1), (3, 1)]],
    "f": [[(4, 9), (6, 9)], [(4, 9), (4, 3), (3, 2), (2, 1)], [(4, 6), (7, 6)]],
    "l": [[(4, 9), (4, 1)]],
    "a": [[(7, 3), (6, 2), (5, 1)], [(7, 3), (7, 6), (6, 7), (5, 7), (4, 6), (3, 5), (2, 3), (2, 2), (3, 1), (4, 1), (5, 2), (6, 3)]],
    "g": [[(8, 3), (7, 2), (6, 1), (5, 1), (4, 2)], [(8, 3), (8, 6), (7, 7), (6, 7), (5, 6), (4, 5), (3, 3), (3, 2), (4, 1), (5, 1), (6, 2)]],
    "t": [[(5, 1), (5, 8), (4, 9), (3, 9)], [(3, 6), (7, 6)]],
    "e": [[(8, 4), (7, 3), (5, 2), (4, 2), (3, 3), (2, 5), (2, 6), (3, 7), (4, 7), (5, 6), (6, 5)], [(8, 4), (2, 4), (8, 4)]],
    "x": [[(1, 9), (8, 1)], [(1, 1), (8, 9)]],
    "_": [[(1, 1), (8, 1)]],
    "0": [[(1, 9), (1, 2), (2, 1), (7, 1), (8, 2), (8, 9), (7, 8), (2, 8), (1, 9)]],
    "1": [[(4, 9), (4, 1), (3, 2)], [(3, 9), (5, 9)]],
    "2": [[(1, 2), (2, 1), (7, 1), (8, 2), (8, 4), (1, 8), (1, 9), (8, 9)]],
    "3": [[(1, 2), (2, 1), (7, 1), (8, 2), (8, 4), (7, 5), (3, 5), (8, 8), (8, 9), (7, 9), (2, 9), (1, 8)]],
    "4": [[(7, 1), (7, 9), (1, 3), (8, 3)]],
    "5": [[(8, 1), (1, 1), (1, 4), (7, 5), (8, 6), (8, 8), (7, 9), (2, 9), (1, 8)]],
    "6": [[(8, 1), (2, 1), (1, 2), (1, 7), (2, 8), (7, 8), (8, 7), (8, 5), (7, 4), (2, 4)]],
    "7": [[(1, 1), (8, 1), (8, 3), (5, 6), (4, 9)]],
    "8": [[(7, 2), (7, 1), (2, 1), (1, 2), (1, 3), (2, 4), (7, 5), (8, 6), (8, 7), (7, 8), (2, 8), (1, 7)]],
    "9": [[(1, 5), (7, 5), (8, 4), (8, 2), (7, 1), (2, 1), (1, 2)]],
    "D": [[(1, 1), (1, 9)], [(1, 9), (7, 9), (8, 8), (8, 2), (7, 1), (1, 1)]],
    "C": [[(8, 2), (7, 1), (2, 1), (1, 2), (1, 8), (2, 9), (7, 9), (8, 8)]],
    "U": [[(1, 9), (1, 3), (2, 2), (7, 2), (8, 3), (8, 9)]],
    "I": [[(3, 1), (6, 1)], [(4, 1), (4, 9)], [(3, 9), (6, 9)]],
    "N": [[(1, 9), (1, 1)], [(1, 1), (8, 9)], [(8, 9), (8, 1)]],
    "S": [[(8, 2), (7, 1), (2, 1), (1, 2), (1, 4), (2, 5), (7, 6), (8, 7), (8, 9), (7, 9), (2, 9), (1, 8)]],
    "P": [[(1, 1), (1, 9)], [(1, 9), (7, 9), (8, 8), (8, 6), (7, 5), (1, 5)]],
    "R": [[(1, 1), (1, 9)], [(1, 9), (7, 9), (8, 8), (8, 6), (7, 5), (1, 5)], [(4, 5), (8, 1)]],
    "M": [[(1, 9), (1, 1)], [(1, 1), (4, 9)], [(4, 9), (8, 1)], [(8, 1), (8, 9)]],
    "W": [[(1, 1), (3, 9), (5, 1), (7, 9), (9, 1)]],
    "E": [[(8, 2), (7, 1), (1, 1), (1, 9), (7, 9), (8, 8)], [(1, 5), (5, 5)]],
    "F": [[(8, 2), (7, 1), (1, 1), (1, 9)], [(1, 5), (6, 5)]],
    "L": [[(1, 9), (1, 1), (8, 1)]],
    "A": [[(1, 1), (4, 9), (8, 1)], [(2, 3), (7, 3)]],
    "V": [[(1, 1), (4, 9), (8, 1)]],
    "X": [[(1, 1), (8, 9)], [(1, 9), (8, 1)]],
    "Y": [[(1, 1), (4, 5), (4, 9)], [(4, 5), (8, 1)]],
    "Z": [[(1, 1), (8, 1), (1, 9), (8, 9)]],
    "K": [[(1, 1), (1, 9)], [(1, 5), (8, 1)], [(3, 5), (8, 9)]],
    "G": [[(8, 2), (7, 1), (2, 1), (1, 2), (1, 8), (2, 9), (7, 9), (8, 8), (8, 5), (5, 5)]],
    "O": [[(1, 2), (2, 1), (7, 1), (8, 2), (8, 8), (7, 9), (2, 9), (1, 8), (1, 2)]],
    "Q": [[(1, 2), (2, 1), (7, 1), (8, 2), (8, 8), (7, 9), (2, 9), (1, 8), (1, 2)], [(5, 5), (8, 1)]],
    "J": [[(1, 2), (2, 1), (6, 1), (8, 2), (8, 9)], [(3, 9), (5, 9)]],
    ".": [[(4, 1), (5, 1)]],
    "/": [[(1, 9), (8, 1)]],
    "-": [[(1, 5), (8, 5)]],
    "=": [[(1, 7), (8, 7)], [(1, 3), (8, 3)]],
    ",": [[(4, 2), (5, 1)]],
    "'": [[(4, 9), (5, 8)]],
    " ": [],
    ":": [[(4, 7), (5, 7)], [(4, 3), (5, 3)]],
    ";": [[(4, 7), (5, 7)], [(4, 3), (5, 2)]],
    "!": [[(4, 9), (4, 3)], [(4, 1), (4, 1)]],
    "?": [[(1, 2), (2, 1), (7, 1), (8, 2), (8, 4), (7, 5), (5, 6), (5, 7)], [(5, 8), (5, 8)]],
    "[": [[(7, 9), (3, 9), (3, 1), (7, 1)]],
    "]": [[(2, 9), (6, 9), (6, 1), (2, 1)]],
    "(": [[(5, 9), (3, 8), (2, 6), (2, 4), (3, 2), (5, 1)]],
    ")": [[(3, 9), (5, 8), (6, 6), (6, 4), (5, 2), (3, 1)]],
    "h": [[(4, 1), (4, 8), (3, 9), (2, 9)], [(4, 6), (7, 6), (8, 5), (8, 1)]],
    "i": [[(4, 9), (5, 9)], [(4, 7), (4, 1)]],
    "j": [[(5, 9), (6, 9)], [(5, 7), (5, 2), (4, 1), (2, 1)], [(3, 1)]],
    "k": [[(4, 1), (4, 9)], [(4, 5), (7, 7)], [(4, 5), (7, 1)]],
    "m": [[(1, 7), (1, 2), (2, 1), (4, 1), (5, 2), (5, 7)], [(5, 2), (6, 1), (8, 1), (9, 2), (9, 7)]],
    "n": [[(1, 7), (1, 2), (2, 1), (4, 1), (5, 2), (5, 7)]],
    "o": [[(2, 2), (3, 1), (6, 1), (7, 2), (7, 6), (6, 7), (3, 7), (2, 6), (2, 2)]],
    "p": [[(1, 1), (1, 7)], [(1, 6), (2, 7), (5, 7), (6, 6), (6, 2), (5, 1), (2, 1), (1, 2)]],
    "q": [[(8, 1), (8, 7)], [(8, 6), (7, 7), (4, 7), (3, 6), (3, 2), (4, 1), (7, 1), (8, 2)]],
    "r": [[(1, 7), (1, 2), (2, 1)], [(1, 5), (5, 5)]],
    "b": [[(1, 1), (1, 9)], [(1, 7), (2, 8), (5, 8), (6, 7), (6, 3), (5, 2), (2, 2), (1, 3)]],
    "c": [[(7, 3), (6, 2), (3, 2), (2, 3), (2, 6), (3, 7), (6, 7), (7, 6)]],
    "d": [[(7, 1), (7, 9)], [(7, 7), (6, 8), (3, 8), (2, 7), (2, 3), (3, 2), (6, 2), (7, 3)]],
    "s": [[(6, 2), (5, 1), (2, 1), (1, 2), (1, 4), (2, 5), (5, 6), (6, 7), (6, 8), (5, 9), (2, 9), (1, 8)]],
    "u": [[(1, 1), (1, 6), (2, 7), (4, 7), (5, 6), (5, 1)]],
    "v": [[(1, 1), (3, 7), (5, 1)]],
    "w": [[(1, 1), (2, 7), (4, 1), (6, 7), (7, 1)]],
    "y": [[(1, 1), (3, 6), (5, 1)], [(5, 6), (6, 7), (8, 7), (9, 6)]],
    "z": [[(1, 1), (6, 1), (1, 7), (6, 7)]],
}


def _render_strokes_ascii(
    strokes: list[list[tuple[float, float]]],
    width: int = 140,
    height: int = 20,
) -> str:
    """Render stroke data at high resolution as ASCII art."""
    if not strokes:
        return "[empty]"
    all_p = [p for s in strokes for p in s]
    if not all_p:
        return "[empty]"
    xs = [p[0] for p in all_p]
    ys = [p[1] for p in all_p]
    mnx, mxx = min(xs), max(xs)
    mny, mxy = min(ys), max(ys)
    w = mxx - mnx or 1
    h = mxy - mny or 1
    aspect = w / h if h else 1
    cols = min(width, int(120 * aspect + 20))
    rows = min(height, max(8, int(cols / aspect * 0.4)))

    grid = [[" "] * cols for _ in range(rows)]

    for stroke in strokes:
        for i in range(len(stroke) - 1):
            x1, y1 = stroke[i]
            x2, y2 = stroke[i + 1]
            seg_len = max(abs(x2 - x1), abs(y2 - y1))
            n = max(int(seg_len / w * cols * 2), 1)
            for t in range(n + 1):
                frac = t / n
                sx = x1 + (x2 - x1) * frac
                sy = y1 + (y2 - y1) * frac
                ix = int((sx - mnx) / w * (cols - 1))
                iy = int((mxy - sy) / h * (rows - 1))
                if 0 <= ix < cols and 0 <= iy < rows:
                    grid[iy][ix] = "#"

    rows_used = [r for r in range(rows) if any(c != " " for c in grid[r])]
    result = []
    for r in rows_used:
        line = "".join(grid[r])
        nc = [(i, c) for i, c in enumerate(line) if c != " "]
        if nc:
            result.append("".join(grid[r][nc[0][0] : nc[-1][0] + 1]))
    return "\n".join(result) if result else "[empty]"


def _analyze_layer_text(path: str, layer_name: str) -> dict:
    """Analyze text content in a Gerber layer using stroke-based analysis."""
    strokes, all_pts = _parse_gerber_strokes(path)
    drawing = [s for s in strokes if len(s) >= 3]
    if not drawing:
        return {"layer": layer_name, "text_groups": []}

    cell_size = 1.0
    cells = defaultdict(list)
    for idx, stroke in enumerate(drawing):
        cx, cy = int(stroke[0][0] / cell_size), int(stroke[0][1] / cell_size)
        cells[(cx, cy)].append(idx)

    unassigned = set(range(len(drawing)))
    text_groups = []
    while unassigned:
        seed = unassigned.pop()
        sx, sy = drawing[seed][0]
        scx, scy = int(sx / cell_size), int(sy / cell_size)
        group = {seed}
        changed = True
        while changed:
            changed = False
            to_add = set()
            for idx in group:
                x, y = drawing[idx][0]
                cx, cy = int(x / cell_size), int(y / cell_size)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for n in cells.get((cx + dx, cy + dy), []):
                            if n in unassigned:
                                to_add.add(n)
            if to_add:
                group.update(to_add)
                unassigned.difference_update(to_add)
                changed = True

        total_pts = sum(len(drawing[i]) for i in group)
        text_groups.append(
            {
                "indices": list(group),
                "strokes": [drawing[i] for i in group],
                "stroke_count": len(group),
                "point_count": total_pts,
            }
        )

    text_groups = [g for g in text_groups if g["point_count"] >= 20]
    text_groups.sort(key=lambda g: -g["point_count"])

    decoded = []
    for g in text_groups[:20]:
        strokes = g["strokes"]
        all_p = [p for s in strokes for p in s]
        xs = [p[0] for p in all_p]
        ys = [p[1] for p in all_p]
        mnx, mxx = min(xs), max(xs)
        mny, mxy = min(ys), max(ys)
        ascii_render = _render_strokes_ascii(strokes, width=100, height=14)
        decoded_text = _decode_text_from_group(strokes)

        entry = {
            "position": (round(mnx, 2), round(mny, 2)),
            "span": (round(mxx - mnx, 3), round(mxy - mny, 3)),
            "stroke_count": g["stroke_count"],
            "point_count": g["point_count"],
            "ascii": ascii_render,
            "decoded": decoded_text,
        }
        decoded.append(entry)

    return {"layer": layer_name, "text_groups": decoded[:15]}


def _segment_strokes_by_column_gap(
    strokes: list[list[tuple[float, float]]],
    column_width: float = 0.02,
    gap_ratio: float = 2.0,
) -> list[list[list[tuple[float, float]]]]:
    """Segment strokes into characters using column-density analysis.

    Divides the x-range into narrow columns, counts points per column,
    then splits where column density drops below a threshold.
    """
    if not strokes:
        return []
    all_p = [p for s in strokes for p in s]
    xs = [p[0] for p in all_p]
    mnx, mxx = min(xs), max(xs)
    width = mxx - mnx
    if width < 0.01:
        return [strokes]

    # Build column density profile
    n_cols = max(int(width / column_width), 20)
    col_density = [0] * n_cols
    for px, _ in all_p:
        col = int((px - mnx) / width * n_cols)
        col = max(0, min(n_cols - 1, col))
        col_density[col] += 1

    # Smooth density (3-col moving average)
    smoothed = [0] * n_cols
    for i in range(n_cols):
        total = 0; cnt = 0
        for di in (-1, 0, 1):
            ni = i + di
            if 0 <= ni < n_cols:
                total += col_density[ni]; cnt += 1
        smoothed[i] = total / max(cnt, 1)

    max_density = max(smoothed) or 1
    threshold = max_density / gap_ratio

    # Find gaps (columns below threshold) and use them as split points
    # A gap must span at least 3 columns
    split_cols = []
    gap_start = None
    for i in range(n_cols):
        if smoothed[i] < threshold:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None and (i - gap_start) >= 3:
                # Split at middle of gap
                split_cols.append((gap_start + i) // 2)
            gap_start = None

    if not split_cols:
        return [strokes]

    # Distribute strokes into character groups using split points
    split_x = [mnx + (col / n_cols) * width for col in split_cols]
    result: list[list[list[tuple[float, float]]]] = []
    remaining = list(strokes)
    for sx in split_x:
        group = [s for s in remaining if min(p[0] for p in s) < sx]
        if group:
            result.append(group)
            remaining = [s for s in remaining if s not in group]
    if remaining:
        result.append(remaining)
    return result


def _decode_text_from_group(
    strokes: list[list[tuple[float, float]]],
) -> str:
    """Decode stroke group as text using point-cloud Hausdorff matching.

    Segments strokes into individual characters by column-density gaps,
    normalizes to 0..9 space, then matches against KiCad Hershey font
    glyphs using symmetric mean nearest-neighbor distance.
    """
    if not strokes:
        return ""

    chars = _segment_strokes_by_column_gap(strokes)

    result = []
    for char_strokes in chars:
        all_p = [p for s in char_strokes for p in s]
        xs = [p[0] for p in all_p]
        ys = [p[1] for p in all_p]
        mnx, mxx = min(xs), max(xs)
        mny, mxy = min(ys), max(ys)
        char_w = mxx - mnx
        char_h = mxy - mny
        if char_w < 0.01 or char_h < 0.01:
            continue

        # Normalize character strokes to 0..9 space
        scale = max(char_w, char_h) / 9.0
        char_pts = [((px - mnx) / scale, (py - mny) / scale) for s in char_strokes for (px, py) in s]

        best_char = "?"
        best_score = 1e9
        for ch, glyph in _KICAD_FONT.items():
            if not glyph:
                continue
            # Font glyphs are already in 0..9 space
            glyph_pts = [p for s in glyph for p in s]

            # Aspect-ratio gate: skip if shapes differ too much
            gxs = [p[0] for p in glyph_pts]; gys = [p[1] for p in glyph_pts]
            gh = max(gys) - min(gys) or 1; gw = max(gxs) - min(gxs) or 1
            char_aspect = char_h / max(char_w, 0.01)
            glyph_aspect = gh / max(gw, 0.01)
            ratio = max(char_aspect, glyph_aspect) / max(min(char_aspect, glyph_aspect), 0.01)
            if ratio > 3.0:
                continue

            # Symmetric mean nearest-neighbor distance
            fwd = sum(
                min(((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                    for gx, gy in glyph_pts)
                for cx, cy in char_pts
            ) / max(len(char_pts), 1)

            rev = sum(
                min(((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                    for cx, cy in char_pts)
                for gx, gy in glyph_pts
            ) / max(len(glyph_pts), 1)

            score = (fwd + rev) / 2
            if score < best_score:
                best_score = score
                best_char = ch

        result.append(best_char if best_score < 1.5 else "?")

    return "".join(result)


def _gerber_to_svg(path: str, layer_name: str = "") -> str:
    """Convert Gerber layer to SVG string."""
    strokes, all_pts = _parse_gerber_strokes(path)
    if not strokes or not all_pts:
        return "" if layer_name else ""

    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x or 1
    h = max_y - min_y or 1
    margin = max(w, h) * 0.05 or 1

    def tx(vx: float) -> float:
        return vx - min_x + margin

    def ty(vy: float) -> float:
        return h - (vy - min_y + margin)

    lines = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        pd = f"M {tx(stroke[0][0]):.4f},{ty(stroke[0][1]):.4f}"
        for sx, sy in stroke[1:]:
            pd += f" L {tx(sx):.4f},{ty(sy):.4f}"
        lines.append(f'<path d="{pd}" fill="none" stroke="black" stroke-width="0.04"/>')

    svg_w = w + 2 * margin
    svg_h = h + 2 * margin
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w:.4f} {svg_h:.4f}"
  width="{svg_w * 40:.0f}" height="{svg_h * 40:.0f}">
  <rect width="100%" height="100%" fill="white"/>
  <g>
    {chr(10).join(lines)}
  </g>
</svg>"""
    return svg


def _pcb_analyze(data: dict) -> dict:
    """Analyze a directory of Gerber/PCB files for CTF challenges."""
    directory = data.get("directory", "")
    search_flag = data.get("search_flag", True)
    render_ascii = data.get("render_ascii", True)
    generate_svg = data.get("generate_svg", False)
    svg_layers = data.get("svg_layers", ["F_Silkscreen", "B_Silkscreen", "F_Fab", "F_Cu"])

    if not directory or not os.path.isdir(directory):
        return {"success": False, "error": f"Directory not found: {directory}"}

    gerber_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".gbr") or f.endswith(".gtl")]
    )
    if not gerber_files:
        gerber_files = sorted(os.listdir(directory))

    results: list[dict] = []
    copper_layers: dict[str, list] = {}
    text_search_hits: list[dict] = []
    svg_outputs: dict[str, str] = {}
    text_analysis: dict[str, dict] = {}

    for fname in gerber_files:
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            continue

        layer_name = fname.replace(".gbr", "").replace("HadesMicro-", "")
        analysis = _analyze_layer(fpath, layer_name)
        results.append(analysis)

        if "Cu" in layer_name or "Cu" in analysis.get("function", ""):
            copper_layers[layer_name] = analysis

        # Generate SVG for requested layers
        if generate_svg:
            layer_short = fname.replace(".gbr", "").replace("HadesMicro-", "")
            if layer_short in svg_layers or "all" in svg_layers:
                svg = _gerber_to_svg(fpath, layer_short)
                if svg:
                    svg_outputs[layer_short] = svg

        # Text analysis via stroke-based detection
        if "Silkscreen" in layer_name or "Fab" in layer_name or "Cu" in layer_name:
            try:
                text_analysis[layer_name] = _analyze_layer_text(fpath, layer_name)
            except Exception:
                logger.debug("pcb_direct: _analyze_layer_text failed for %s", layer_name, exc_info=True)

        # Search for flag patterns in raw content
        if search_flag:
            with open(fpath, errors="replace") as f:
                raw = f.read()
            # Search for HTB{...} or similar flag patterns
            flag_matches = re.findall(r"[A-Z]{3,4}\{[^}]+\}", raw)
            for fm in flag_matches:
                text_search_hits.append({"file": fname, "flag_candidate": fm})

    # Layer comparison for copper layers
    via_positions = []
    if len(copper_layers) >= 2:
        copper_draw = {}
        for name, a in copper_layers.items():
            if "draw_coords" in a:
                coords_data = _parse_gerber_coords(
                    os.path.join(directory, gerber_files[0])
                )
                copper_draw[name] = a
        # Use draw counts as simple comparison
        draw_counts = {n: a.get("draw_count", 0) for n, a in copper_layers.items()}

    summary = {
        "success": True,
        "directory": directory,
        "files_found": len(gerber_files),
        "layers_analyzed": len(results),
        "layer_names": [r["layer"] for r in results],
        "board_size": None,
        "copper_layer_count": len(copper_layers),
        "text_search_hits": text_search_hits[:10] if text_search_hits else [],
        "flag_found": bool(text_search_hits),
    }

    # Board size from Edge_Cuts
    for r in results:
        if "Edge" in r["layer"] or "edge" in r.get("function", "").lower():
            if "x_span" in r and "y_span" in r:
                summary["board_size"] = (r["x_span"], r["y_span"])

    if render_ascii:
        summary["ascii_art"] = [
            r["ascii_art"]
            for r in results
            if "ascii_art" in r
            and ("Silkscreen" in r["layer"] or "Fab" in r["layer"])
        ][:3]

    if generate_svg and svg_outputs:
        summary["svg_count"] = len(svg_outputs)
        summary["svg_layers"] = list(svg_outputs.keys())
        # Write SVGs to directory
        svg_dir = os.path.join(directory, "_svg")
        os.makedirs(svg_dir, exist_ok=True)
        written = []
        for name, svg in svg_outputs.items():
            fpath = os.path.join(svg_dir, f"{name}.svg")
            with open(fpath, "w") as f:
                f.write(svg)
            written.append(fpath)
        summary["svg_files"] = written

    if text_analysis:
        summary["text_regions"] = {
            layer: ta["text_groups"]
            for layer, ta in text_analysis.items()
            if ta.get("text_groups")
        }

    summary["layers"] = results

    return summary


_HANDLERS = {
    "pcb_analyze": _pcb_analyze,
}


def pcb_exec(tool: str, data: dict) -> dict:
    handler = _HANDLERS.get(tool)
    if handler is None:
        return {"success": False, "error": f"Unknown pcb tool: '{tool}'"}
    return handler(data)
