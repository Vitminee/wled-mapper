"""Local web server that exposes mapping controls and conversion helpers."""

import csv
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.map_leds import (
    MappingConfig,
    MappingError,
    MappingResult,
    run_mapping,
)

from scripts.convert_mapping import (
    build_grid,
    load_mapping,
)


DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_CSV = DATA_DIR / "mapping.csv"

CAMERA_SCAN_LIMIT = 8


app = FastAPI(title="WLED Mapper Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

app.mount("/data", StaticFiles(directory="data"), name="data")

app.mount("/scripts", StaticFiles(directory="scripts"), name="scripts")


@app.get("/")
def root():

    return RedirectResponse("/ui/led_viewer.html")


class MapRequest(BaseModel):

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


class ConvertRequest(BaseModel):

    step: Optional[float] = 25.0

    width: Optional[int] = None

    height: Optional[int] = None

    include_meta: bool = True


STATE = {
    "status": "idle",
    "message": "Ready",
    "current": 0,
    "total": 0,
    "started_at": None,
    "finished_at": None,
    "points_captured": 0,
}


STATE_LOCK = threading.Lock()

RESULT: Optional[MappingResult] = None
LIVE_POINTS: List[Dict[str, float]] = []
STOP_EVENT = threading.Event()


def _reset_mapping_csv():

    with MAPPING_CSV.open("w", newline="", encoding="ascii") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(["led", "x", "y", "brightness"])


def _append_mapping_entry(entry: Dict[str, float]):

    new_file = not MAPPING_CSV.exists() or MAPPING_CSV.stat().st_size == 0

    with MAPPING_CSV.open("a", newline="", encoding="ascii") as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=["led", "x", "y", "brightness"])

        if new_file:

            writer.writeheader()

        writer.writerow(entry)


def _try_open_camera(camera_index: int) -> Optional[cv2.VideoCapture]:

    if sys.platform.startswith("win"):

        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]

    else:

        backends = [cv2.CAP_ANY]

    for backend in backends:

        try:

            cap = cv2.VideoCapture(camera_index, backend)

        except Exception:  # pylint: disable=broad-except

            continue

        if cap.isOpened():

            return cap

        cap.release()

    return None


def _list_available_cameras(max_index: int) -> list[int]:

    cameras: list[int] = []

    for idx in range(max(1, max_index)):

        cap = _try_open_camera(idx)

        if cap is None:

            continue

        cap.release()

        cameras.append(idx)

    return cameras


def _update_state(**kwargs):

    with STATE_LOCK:

        STATE.update(kwargs)


def _progress_hook(led_index: int, stage: str, payload):

    if stage == "captured" and payload:

        point = {
            "led": int(payload.get("led", led_index)),
            "x": float(payload.get("x", 0.0)),
            "y": float(payload.get("y", 0.0)),
            "brightness": float(payload.get("brightness", 0.0)),
        }

        with STATE_LOCK:

            LIVE_POINTS.append(point)

            STATE.update(
                {
                    "status": "running",
                    "message": "captured",
                    "current": led_index + 1,
                    "points_captured": len(LIVE_POINTS),
                }
            )

            _append_mapping_entry(point)

        return

    if stage == "skipped":

        message = "skipped"
        if payload and payload.get("reason"):
            message = payload["reason"]

        _update_state(
            status="running",
            message=message,
            current=max(0, led_index + 1),
        )

        return

    if led_index < 0:

        return

    _update_state(
        status="running",
        message=stage,
        current=led_index + (0 if stage == "highlight" else 0.5),
    )


def _mapping_thread(config: MappingConfig, stop_event: threading.Event):

    global RESULT

    try:

        _update_state(
            status="running",
            message="Capturing",
            current=0,
            total=config.led_count,
            started_at=time.time(),
        )

        result = run_mapping(config, hook=_progress_hook, stop_event=stop_event)

        RESULT = result

        status = "completed"
        message = "Capture finished"
        current = config.led_count
        if result.stopped:
            status = "stopped"
            message = "Capture stopped"
            current = len(result.entries)

        _update_state(
            status=status,
            message=message,
            finished_at=time.time(),
            current=current,
        )

    except (MappingError, Exception) as exc:  # pylint: disable=broad-except

        RESULT = None

        _update_state(status="error", message=str(exc), finished_at=time.time())
    finally:
        stop_event.clear()


@app.post("/api/map")
def start_mapping(req: MapRequest):

    with STATE_LOCK:

        if STATE["status"] == "running":

            raise HTTPException(status_code=409, detail="Mapping already in progress")

        STATE.update(
            {
                "status": "starting",
                "message": "Launching capture...",
                "current": 0,
                "total": req.led_count,
                "started_at": None,
                "finished_at": None,
                "points_captured": 0,
            }
        )

        LIVE_POINTS.clear()
        _reset_mapping_csv()
        STOP_EVENT.clear()

    config = MappingConfig(**req.model_dump())

    thread = threading.Thread(
        target=_mapping_thread, args=(config, STOP_EVENT), daemon=True
    )

    thread.start()

    return {"ok": True}


@app.get("/api/status")
def get_status():

    with STATE_LOCK:

        data = STATE.copy()

    return data


@app.get("/api/cameras")
def get_cameras(max_index: int = CAMERA_SCAN_LIMIT):

    limit = max(1, min(max_index, 24))

    cameras = _list_available_cameras(limit)

    return {"cameras": [{"index": idx} for idx in cameras], "scanned": limit}


@app.get("/api/camera_preview/{camera_index}")
def camera_preview(camera_index: int, width: int = 640):

    cap = _try_open_camera(camera_index)

    if cap is None:

        raise HTTPException(status_code=404, detail="Camera not available")

    success, frame = cap.read()

    cap.release()

    if not success or frame is None:

        raise HTTPException(status_code=500, detail="Failed to capture frame")

    target_width = max(160, min(width, 1280))

    height, width_px = frame.shape[:2]

    if width_px != target_width:

        scale = target_width / max(1, width_px)

        new_height = max(1, int(height * scale))

        frame = cv2.resize(frame, (target_width, new_height))

    ok, encoded = cv2.imencode(".jpg", frame)

    if not ok:

        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/live_points")
def get_live_points(after: int = -1):

    with STATE_LOCK:

        total = len(LIVE_POINTS)

        if after >= 0:

            points = [pt for pt in LIVE_POINTS if pt["led"] > after]

        else:

            points = list(LIVE_POINTS)

    return {"points": points, "total": total}


@app.post("/api/stop")
def stop_mapping():

    with STATE_LOCK:

        if STATE["status"] not in {"running", "starting"}:

            raise HTTPException(status_code=409, detail="No mapping in progress")

    STOP_EVENT.set()

    return {"ok": True}


@app.get("/api/result")
def get_result():

    if RESULT is None:

        raise HTTPException(status_code=404, detail="No capture result available")

    return {"entries": RESULT.entries}


@app.post("/api/convert")
def convert(req: ConvertRequest):

    if not MAPPING_CSV.exists():

        raise HTTPException(
            status_code=404, detail="mapping.csv not found, run capture first"
        )

    rows = load_mapping(MAPPING_CSV)

    if req.step is None and req.width is None and req.height is None:

        raise HTTPException(status_code=400, detail="Provide step or width/height")

    effective_step = req.step if req.step is not None else 10.0

    grid = build_grid(rows, effective_step, req.width, req.height)

    # Do not write ledmap.json on server; UI handles download formatting

    payload = {
        "map": grid["map"],
        "width": grid["width"],
        "height": grid["height"],
        "meta": grid["meta"],
    }

    return payload


@app.post("/api/gaps")
def generate_gaps(req: ConvertRequest):

    data = convert(req)

    gap_map = [1 if value != -1 else -1 for value in data["map"]]

    # Do not write 2d-gaps.json on server; UI handles download formatting
    return gap_map


if __name__ == "__main__":

    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
