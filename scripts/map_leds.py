#!/usr/bin/env python3
"""Automated LED-to-camera mapping using the WLED HTTP API.

Exposes a reusable `run_mapping` function so other modules (e.g. the web
server) can drive the capture pipeline, while still supporting CLI usage.
"""
import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import requests


ProgressHook = Callable[[int, str, Dict[str, float]], None]

LOGGER = logging.getLogger("wled.mapper")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


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
    skip_dim_leds: bool = False
    timeout: float = 3.0
    sleep_every: int = 25
    cooldown: float = 2.0


@dataclass
class MappingResult:
    entries: List[Dict[str, float]]
    output_path: Optional[Path] = None
    stopped: bool = False


class MappingError(RuntimeError):
    """Raised when the mapping pipeline fails."""


class DimBrightnessError(MappingError):
    """Raised when the brightest pixel does not exceed the minimum threshold."""


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
        "--skip-dim-leds",
        action="store_true",
        help="Skip LEDs that never exceed --min-brightness instead of failing the run",
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


OFF_HEX = "000000"
ON_HEX = "FFFFFF"
LED_RETRY_ATTEMPTS = 5


def build_state_url(host: str) -> str:
    return host.rstrip("/") + "/json/state"


def _build_range_payload(led_count: int, active_led: Optional[int]) -> List[object]:
    if active_led is None:
        return [0, led_count, OFF_HEX]

    ranges: List[object] = []
    if active_led > 0:
        ranges.extend([0, active_led, OFF_HEX])
    ranges.extend([active_led, active_led + 1, ON_HEX])
    if active_led + 1 < led_count:
        ranges.extend([active_led + 1, led_count, OFF_HEX])
    return ranges


def _log_payload(label: str, payload: Dict[str, object]) -> None:
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "WLED POST [%s]: %s", label, json.dumps(payload, ensure_ascii=False)
        )


def initialize_segment(
    state_url: str,
    led_count: int,
    timeout: float,
    segment_index: int,
    hook: Optional[ProgressHook],
) -> None:
    payload: Dict[str, object] = {
        "on": True,
        "seg": [
            {
                "id": segment_index,
                "start": 0,
                "stop": led_count,
                "fx": 0,
                "pal": 0,
            }
        ],
    }
    _log_payload("initialize_segment", payload)
    response = requests.post(state_url, json=payload, timeout=timeout)
    response.raise_for_status()
    if hook:
        hook(-1, "init", {"payload": payload})


def send_active_led(
    state_url: str,
    led_count: int,
    active_led: Optional[int],
    timeout: float,
    label: str,
    hook: Optional[ProgressHook],
    segment_index: int,
) -> None:
    ranges = _build_range_payload(led_count, active_led)
    payload: Dict[str, object] = {
        "seg": [
            {
                "id": segment_index,
                "i": ranges,
            }
        ]
    }
    _log_payload(label, payload)
    response = requests.post(state_url, json=payload, timeout=timeout)
    response.raise_for_status()
    if hook:
        hook(-1 if active_led is None else active_led, label, {"range_entries": len(ranges)})


def collect_brightest_points(
    cap: cv2.VideoCapture, camera_index: int, frames: int
) -> Tuple[cv2.VideoCapture, List[Tuple[float, Tuple[int, int]]]]:
    readings: List[Tuple[float, Tuple[int, int]]] = []
    current_cap = cap
    for _ in range(frames):
        current_cap, frame = _read_frame(current_cap, camera_index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        readings.append((max_val, max_loc))
        time.sleep(0.05)
    return current_cap, readings


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
        raise DimBrightnessError(
            f"Brightness {peak_val:.2f} below threshold {min_brightness}."
        )
    return avg_x, avg_y, peak_val


def _open_camera(camera_index: int, attempts: int = 3, delay: float = 0.2) -> cv2.VideoCapture:
    backends: List[int] = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    last_exc: Optional[MappingError] = None

    for _ in range(max(1, attempts)):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(camera_index, backend)
            except Exception:  # pylint: disable=broad-except
                continue
            if cap is not None and cap.isOpened():
                return cap
            cap.release()
        last_exc = MappingError(
            "Unable to open camera. Check --camera-index and permissions."
        )
        time.sleep(delay)

    if last_exc:
        raise last_exc
    raise MappingError("Unable to open camera. Check --camera-index and permissions.")



def _read_frame(cap: cv2.VideoCapture, camera_index: int) -> Tuple[cv2.VideoCapture, Any]:
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap, frame
    cap.release()
    cap = _open_camera(camera_index)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise MappingError("Failed to grab frame from camera.")
    return cap, frame



def run_mapping(
    config: MappingConfig,
    hook: Optional[ProgressHook] = None,
    stop_event: Optional[Event] = None,
) -> MappingResult:
    state_url = build_state_url(config.host)
    cap = _open_camera(config.camera_index)
    results: List[Dict[str, float]] = []
    stopped = False
    try:
        initialize_segment(
            state_url,
            config.led_count,
            config.timeout,
            config.segment_index,
            hook,
        )
        send_active_led(
            state_url,
            config.led_count,
            None,
            config.timeout,
            "initial-strip",
            hook,
            segment_index=config.segment_index,
        )
        if config.prelight_delay > 0:
            time.sleep(config.prelight_delay)
        for led in range(config.led_count):
            if stop_event and stop_event.is_set():
                stopped = True
                break
            attempt = 0
            while True:
                attempt += 1
                send_active_led(
                    state_url,
                    config.led_count,
                    led,
                    config.timeout,
                    "highlight",
                    hook,
                    segment_index=config.segment_index,
                )
                time.sleep(config.capture_delay)
                try:
                    cap, readings = collect_brightest_points(
                        cap, config.camera_index, config.frames_per_led
                    )
                    x, y, peak = compute_coordinate(
                        readings, config.top_frame_ratio, config.min_brightness
                    )
                    point = {
                        "led": led,
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "brightness": round(peak, 2),
                    }
                    results.append(point)
                    if hook:
                        hook(led, "captured", point)
                    break
                except DimBrightnessError as exc:
                    LOGGER.warning(
                        "LED %s capture failed (attempt %s/%s): %s",
                        led,
                        attempt,
                        LED_RETRY_ATTEMPTS,
                        exc,
                    )
                    if config.skip_dim_leds:
                        LOGGER.info("Skipping LED %s due to low brightness", led)
                        if hook:
                            hook(
                                led,
                                "skipped",
                                {"attempt": attempt, "reason": str(exc)},
                            )
                        break
                    if hook:
                        hook(
                            led,
                            "retry",
                            {"attempt": attempt, "max": LED_RETRY_ATTEMPTS, "reason": str(exc)},
                        )
                    if attempt >= LED_RETRY_ATTEMPTS:
                        raise
                    time.sleep(max(config.prelight_delay, 0.05))
                    continue
                except MappingError as exc:
                    LOGGER.warning(
                        "LED %s capture failed (attempt %s/%s): %s",
                        led,
                        attempt,
                        LED_RETRY_ATTEMPTS,
                        exc,
                    )
                    if hook:
                        hook(
                            led,
                            "retry",
                            {"attempt": attempt, "max": LED_RETRY_ATTEMPTS, "reason": str(exc)},
                        )
                    if attempt >= LED_RETRY_ATTEMPTS:
                        raise
                    time.sleep(max(config.prelight_delay, 0.05))
                    continue
            if stop_event and stop_event.is_set():
                stopped = True
                break
            if config.postlight_delay > 0:
                time.sleep(config.postlight_delay)
            if config.prelight_delay > 0 and led != config.led_count - 1:
                time.sleep(config.prelight_delay)
            if config.sleep_every and (led + 1) % config.sleep_every == 0:
                time.sleep(config.cooldown)
    finally:
        cap.release()
        try:
            send_active_led(
                state_url,
                config.led_count,
                None,
                config.timeout,
                "final-reset",
                hook,
                segment_index=config.segment_index,
            )
        except requests.RequestException:
            pass
    return MappingResult(entries=results, stopped=stopped)


def write_csv(entries: Iterable[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["led", "x", "y", "brightness"])
        writer.writeheader()
        writer.writerows(entries)


def main() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
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
        skip_dim_leds=args.skip_dim_leds,
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
