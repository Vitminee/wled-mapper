#!/usr/bin/env python3

"""Convert camera-derived mapping CSV into a grid-based LED map JSON."""

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def format_led_map(values: List[int], width: int, height: int) -> str:
    lines = ['{"map":', "  ["]
    for row in range(height):
        start = row * width
        row_values = values[start : start + width]
        formatted = ", ".join(f"{value:2d}" for value in row_values)
        suffix = "," if row < height - 1 else ""
        lines.append(f"    {formatted}{suffix}")
    lines.append("  ],")
    lines.append(f'  "width":  {width},')
    lines.append(f'  "height": {height}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def format_gap_map(values: List[int], width: int, height: int) -> str:
    lines = ["["]
    for row in range(height):
        start = row * width
        row_values = values[start : start + width]
        suffix = "," if row < height - 1 else ""
        lines.append("    " + ", ".join(f"{value:2d}" for value in row_values) + suffix)
    lines.append("]")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Convert mapping CSV (led,x,y,brightness) into a discrete grid JSON map.",
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/mapping.csv"),
        help="CSV file produced by map_leds.py (default: data/mapping.csv)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ledmap.json"),
        help="Where to write the JSON map (default: ledmap.json)",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Quantization step in camera units. Ignored for any axis where a target size is given.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Target grid width (columns). When set, coordinates are stretched to this width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target grid height (rows). When set, coordinates are stretched to this height.",
    )

    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include origin/span metadata in the generated JSON.",
    )

    return parser.parse_args()


def load_mapping(csv_path: Path) -> List[Dict[str, float]]:

    with csv_path.open("r", newline="", encoding="ascii") as handle:

        reader = csv.DictReader(handle)

        rows: List[Dict[str, float]] = []

        for row in reader:

            rows.append(
                {
                    "led": int(row["led"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "brightness": float(row.get("brightness", 0)),
                }
            )

    if not rows:

        raise ValueError(f"No records in {csv_path}")

    return rows


def _scale_for_axis(
    min_val: float, max_val: float, target: Optional[int], step: float
) -> Tuple[int, float]:

    if target is not None:

        if target <= 0:

            raise ValueError("width/height must be positive")

        if max_val == min_val:

            return target, 0.0

        return target, (target - 1) / (max_val - min_val)

    if step <= 0:

        raise ValueError("--step must be positive when width/height are not provided")

    if max_val == min_val:

        return 1, 0.0

    span = max_val - min_val

    scale = 1.0 / step

    size = max(1, int(round(span * scale)) + 1)

    return size, scale


def _iter_slots(width: int, height: int, start_x: int, start_y: int):

    seen = [[False] * width for _ in range(height)]

    queue: deque[Tuple[int, int]] = deque()

    if 0 <= start_x < width and 0 <= start_y < height:

        queue.append((start_x, start_y))

        seen[start_y][start_x] = True

    else:

        for y in range(height):

            for x in range(width):

                queue.append((x, y))

        while queue:

            yield queue.popleft()

        return

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:

        cx, cy = queue.popleft()

        yield cx, cy

        for dx, dy in directions:

            nx, ny = cx + dx, cy + dy

            if 0 <= nx < width and 0 <= ny < height and not seen[ny][nx]:

                seen[ny][nx] = True

                queue.append((nx, ny))

    for y in range(height):

        for x in range(width):

            if not seen[y][x]:

                yield x, y


def _trim_grid(grid: List[List[int]]) -> Tuple[List[List[int]], int, int]:

    if not grid or not grid[0]:

        return grid, 0, 0

    height = len(grid)

    width = len(grid[0])

    top = 0

    while top < height and all(cell == -1 for cell in grid[top]):

        top += 1

    bottom = height - 1

    while bottom >= top and all(cell == -1 for cell in grid[bottom]):

        bottom -= 1

    left = 0

    while left < width and all(row[left] == -1 for row in grid):

        left += 1

    right = width - 1

    while right >= left and all(row[right] == -1 for row in grid):

        right -= 1

    if top > bottom or left > right:

        return [[-1]], 0, 0

    trimmed = [row[left : right + 1] for row in grid[top : bottom + 1]]

    return trimmed, left, top


def build_grid(
    rows: List[Dict[str, float]],
    step: float,
    target_width: Optional[int],
    target_height: Optional[int],
) -> Dict[str, object]:

    min_x = min(row["x"] for row in rows)

    max_x = max(row["x"] for row in rows)

    min_y = min(row["y"] for row in rows)

    max_y = max(row["y"] for row in rows)

    working_step = step

    attempt = 0

    while True:

        width, x_scale = _scale_for_axis(min_x, max_x, target_width, working_step)

        height, y_scale = _scale_for_axis(min_y, max_y, target_height, working_step)

        grid = [[-1 for _ in range(width)] for _ in range(height)]

        try:

            for row in rows:

                led = row["led"]

                x_idx = (
                    0 if x_scale == 0.0 else int(round((row["x"] - min_x) * x_scale))
                )

                y_idx = (
                    0 if y_scale == 0.0 else int(round((row["y"] - min_y) * y_scale))
                )

                x_idx = max(0, min(width - 1, x_idx))

                y_idx = max(0, min(height - 1, y_idx))

                for nx, ny in _iter_slots(width, height, x_idx, y_idx):

                    if grid[ny][nx] in (-1, led):

                        grid[ny][nx] = led

                        break

                else:

                    raise ValueError(f"Unable to place LED {led}; grid saturated")

        except ValueError as error:

            if target_width or target_height:

                raise

            working_step *= 0.7

            attempt += 1

            if working_step < 0.05 or attempt > 8:

                raise error

            continue

        break

    trimmed_grid, offset_left, offset_top = _trim_grid(grid)

    final_height = len(trimmed_grid)

    final_width = len(trimmed_grid[0]) if final_height else 0

    flat_map: List[int] = []

    for row in trimmed_grid:

        flat_map.extend(row)

    metadata = {
        "origin": {"x": min_x, "y": min_y},
        "span": {"x": max_x - min_x, "y": max_y - min_y},
        "scale": {
            "x_per_cell": (
                0 if final_width <= 1 else (max_x - min_x) / (final_width - 1)
            ),
            "y_per_cell": (
                0 if final_height <= 1 else (max_y - min_y) / (final_height - 1)
            ),
        },
        "mode": {
            "width": target_width is not None,
            "height": target_height is not None,
            "step": step,
            "effective_step": working_step,
        },
        "trim_offset": {"x": offset_left, "y": offset_top},
    }

    return {
        "map": flat_map,
        "width": final_width,
        "height": final_height,
        "meta": metadata,
    }


def main() -> None:

    args = parse_args()

    rows = load_mapping(args.input)

    grid = build_grid(rows, args.step, args.width, args.height)

    args.output.write_text(
        format_led_map(grid["map"], grid["width"], grid["height"]), encoding="ascii"
    )

    print(
        "Wrote {} with grid {}x{}".format(
            args.output,
            grid["width"],
            grid["height"],
        )
    )


if __name__ == "__main__":

    main()
