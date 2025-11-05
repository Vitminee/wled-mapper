"""Local web server that exposes mapping controls and conversion helpers."""

import threading

import time

from pathlib import Path

from typing import Optional


from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import RedirectResponse

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel


from scripts.map_leds import (
    MappingConfig,
    MappingError,
    MappingResult,
    run_mapping,
    write_csv,
)

from scripts.convert_mapping import (
    build_grid,
    load_mapping,
)


DATA_DIR = Path("data")

MAPPING_CSV = DATA_DIR / "mapping.csv"


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
}


STATE_LOCK = threading.Lock()

RESULT: Optional[MappingResult] = None


def _update_state(**kwargs):

    with STATE_LOCK:

        STATE.update(kwargs)


def _progress_hook(led_index: int, stage: str, _payload):

    if led_index < 0:

        return

    _update_state(
        status="running",
        message=stage,
        current=led_index + (0 if stage == "highlight" else 0.5),
    )


def _mapping_thread(config: MappingConfig):

    global RESULT

    try:

        _update_state(
            status="running",
            message="Capturing",
            current=0,
            total=config.led_count,
            started_at=time.time(),
        )

        result = run_mapping(config, hook=_progress_hook)

        RESULT = result

        write_csv(result.entries, MAPPING_CSV)

        _update_state(
            status="completed",
            message="Capture finished",
            finished_at=time.time(),
            current=config.led_count,
        )

    except (MappingError, Exception) as exc:  # pylint: disable=broad-except

        RESULT = None

        _update_state(status="error", message=str(exc), finished_at=time.time())


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
            }
        )

    config = MappingConfig(**req.model_dump())

    thread = threading.Thread(target=_mapping_thread, args=(config,), daemon=True)

    thread.start()

    return {"ok": True}


@app.get("/api/status")
def get_status():

    with STATE_LOCK:

        data = STATE.copy()

    return data


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
