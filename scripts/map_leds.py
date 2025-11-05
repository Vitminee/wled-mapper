#!/usr/bin/env python3
"""Automated LED-to-camera mapping using the WLED HTTP API.

Exposes a reusable `run_mapping` function so other modules (e.g. the web
server) can drive the capture pipeline, while still supporting CLI usage.
"""
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import requests


ProgressHook = Callable[[int, str, Dict[str, float]], None]


@dataclass
class MappingConfig:
    host: str
    led_count: int
    camera_index: int = 0
    segment_index: int = 0
    transition_ms: int = 0
    prelight_delay: float = 0.3
    capture_delay: float = 0.6
    postlight_delay: float = 0.2
    frames_per_led: int = 5
    top_frame_ratio: float = 0.4
    min_brightness: float = 30.0
    timeout: float = 3.0
    sleep_every: int = 25
    cooldown: float = 2.0


@dataclass
class MappingResult:
    entries: List[Dict[str, float]]
    output_path: Optional[Path] = None


class MappingError(RuntimeError):
    """Raised when the mapping pipeline fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively lights each LED via the WLED HTTP API, captures the brightest camera point, and stores the LED-to-camera mapping.",
    )
    parser.add_argument(
        "host", help="Base URL or IP of the WLED controller, e.g., http://192.168.1.120"
    )
    parser.add_argument("led_count", type=int, help="Total number of LEDs to map")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index passed to OpenCV VideoCapture (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("led_mapping.csv"),
        help="Where to store the resulting CSV mapping (default: led_mapping.csv)",
    )
    parser.add_argument(
        "--prelight-delay",
        type=float,
        default=0.1,
        help="Delay between reset and highlight requests (default: 0.1)",
    )
    parser.add_argument(
        "--capture-delay",
        type=float,
        default=0.4,
        help="Delay in seconds after setting an LED before capturing (default: 0.4)",
    )
    parser.add_argument(
        "--postlight-delay",
        type=float,
        default=0.1,
        help="Delay after capture before resetting LEDs (default: 0.1)",
    )
    parser.add_argument(
        "--frames-per-led",
        type=int,
        default=5,
        help="Number of frames to sample per LED (default: 5)",
    )
    parser.add_argument(
        "--top-frame-ratio",
        type=float,
        default=0.4,
        help="Fraction of brightest frames to average for the final coordinate (default: 0.4)",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=30.0,
        help="Minimum grayscale value required to accept a blob (default: 30.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for the WLED HTTP request (default: 3.0)",
    )
    parser.add_argument(
        "--sleep-every",
        type=int,
        default=25,
        help="Pause after this many LEDs to avoid overheating (default: 25)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown duration in seconds (default: 2.0)",
    )
    return parser.parse_args()


def build_win_url(host: str) -> str:
    return host.rstrip("/") + "/win"


def build_query_url(win_url: str, params: Dict[str, int]) -> str:
    query = "".join(f"&{key}={value}" for key, value in params.items())
    return win_url + query


def reset_strip(
    win_url: str,
    led_count: int,
    timeout: float,
    label: str,
    hook: Optional[ProgressHook],
    led_index: int = -1,
    segment_index: int = 0,
    transition_ms: int = 0,
) -> None:
    # Select segment, cover full range, set RGB=0 and brightness A=0 to ensure all LEDs are dark
    params = {
        "SS": segment_index,
        "S": 0,
        "S2": led_count,
        "R": 0,
        "G": 0,
        "B": 0,
        "A": 0,
        "TT": max(0, int(transition_ms)),
    }
    url = build_query_url(win_url, params)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if hook:
        hook(led_index, label, {"url": url})


def highlight_led(
    win_url: str,
    led_index: int,
    timeout: float,
    hook: Optional[ProgressHook],
    segment_index: int = 0,
    transition_ms: int = 0,
) -> None:
    # Ensure segment is selected and brightness is sufficient
    params = {
        "SS": segment_index,
        "S": led_index,
        "S2": led_index + 1,
        "R": 255,
        "G": 255,
        "B": 255,
        "A": 255,
        "TT": max(0, int(transition_ms)),
    }
    url = build_query_url(win_url, params)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if hook:
        hook(led_index, "highlight", {"url": url})


def collect_brightest_points(
    cap: cv2.VideoCapture, frames: int
) -> List[Tuple[float, Tuple[int, int]]]:
    readings: List[Tuple[float, Tuple[int, int]]] = []
    for _ in range(frames):
        if not cap.isOpened():
            raise MappingError("Camera stream is not opened.")
        ret, frame = cap.read()
        if not ret or frame is None:
            raise MappingError("Failed to grab frame from camera.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        readings.append((max_val, max_loc))
        time.sleep(0.05)
    return readings


def compute_coordinate(
    readings: List[Tuple[float, Tuple[int, int]]], ratio: float, min_brightness: float
) -> Tuple[float, float, float]:
    if not readings:
        raise MappingError("No readings captured for LED.")
    readings.sort(key=lambda item: item[0], reverse=True)
    take = max(1, int(len(readings) * ratio))
    top = readings[:take]
    avg_x = sum(loc[0] for _, loc in top) / take
    avg_y = sum(loc[1] for _, loc in top) / take
    peak_val = top[0][0]
    if peak_val < min_brightness:
        raise MappingError(
            f"Brightness {peak_val:.2f} below threshold {min_brightness}."
        )
    return avg_x, avg_y, peak_val


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise MappingError(
            "Unable to open camera. Check --camera-index and permissions."
        )
    return cap


def run_mapping(
    config: MappingConfig, hook: Optional[ProgressHook] = None
) -> MappingResult:
    win_url = build_win_url(config.host)
    cap = _open_camera(config.camera_index)
    results: List[Dict[str, float]] = []
    try:
        # Ensure strip is dark before starting
        reset_strip(
            win_url,
            config.led_count,
            config.timeout,
            "initial-reset",
            hook,
            segment_index=config.segment_index,
            transition_ms=config.transition_ms,
        )
        if config.prelight_delay > 0:
            time.sleep(config.prelight_delay)
        for led in range(config.led_count):
            # Always turn everything off first, then light one LED
            reset_strip(
                win_url,
                config.led_count,
                config.timeout,
                "pre-highlight-reset",
                hook,
                led,
                segment_index=config.segment_index,
                transition_ms=config.transition_ms,
            )
            if config.prelight_delay > 0:
                time.sleep(config.prelight_delay)
            highlight_led(
                win_url,
                led,
                config.timeout,
                hook,
                segment_index=config.segment_index,
                transition_ms=config.transition_ms,
            )
            time.sleep(config.capture_delay)
            readings = collect_brightest_points(cap, config.frames_per_led)
            x, y, peak = compute_coordinate(
                readings, config.top_frame_ratio, config.min_brightness
            )
            results.append(
                {
                    "led": led,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "brightness": round(peak, 2),
                }
            )
            if config.postlight_delay > 0:
                time.sleep(config.postlight_delay)
            # Turn everything off again after capture
            reset_strip(
                win_url,
                config.led_count,
                config.timeout,
                "per-led-reset",
                hook,
                led,
                segment_index=config.segment_index,
                transition_ms=config.transition_ms,
            )
            if config.prelight_delay > 0 and led != config.led_count - 1:
                time.sleep(config.prelight_delay)
            if config.sleep_every and (led + 1) % config.sleep_every == 0:
                time.sleep(config.cooldown)
    finally:
        cap.release()
    return MappingResult(entries=results)


def write_csv(entries: Iterable[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["led", "x", "y", "brightness"])
        writer.writeheader()
        writer.writerows(entries)


def main() -> None:
    args = parse_args()
    config = MappingConfig(
        host=args.host,
        led_count=args.led_count,
        camera_index=args.camera_index,
        prelight_delay=args.prelight_delay,
        capture_delay=args.capture_delay,
        postlight_delay=args.postlight_delay,
        frames_per_led=args.frames_per_led,
        top_frame_ratio=args.top_frame_ratio,
        min_brightness=args.min_brightness,
        timeout=args.timeout,
        sleep_every=args.sleep_every,
        cooldown=args.cooldown,
    )
    try:
        result = run_mapping(config)
    except (requests.RequestException, MappingError) as exc:
        sys.exit(str(exc))
    write_csv(result.entries, args.output)
    print(f"Recorded {len(result.entries)} LED positions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
