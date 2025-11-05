# WLED Mapper

This project lets you capture LED positions from a camera, adjust the layout, and export both the LED map and a 2D gap mask.

## Prerequisites
- Python 3.10+
- Camera accessible to the host machine
- WLED controller reachable via HTTP on your network

## Setup
```bash
python -m venv .venv
.venv\\Scripts\\activate  # on Windows
pip install -r requirements.txt
```

## Running the app
```bash
python server.py
```
Open `http://localhost:8000/` in your browser. The UI provides:
- Mapping controls (WLED host, LED count, segment, camera index, frames, brightness)
- Timing controls (prelight/capture/postlight delays) and WLED transition (`TT` in ms)
- Live progress display while mapping runs
- Grid preview & camera scatter plot
- Buttons to generate/download `ledmap.json` and `2d-gaps.json` (client-side download)

## Usage Guide

### 1) Prepare WLED
- Ensure the controller is reachable (open its UI, e.g., `http://10.0.0.50/`).
- Use a single segment covering the entire strip (start 0, length = LED count).
- Use 1D LED layout in WLED (disable 2D matrix/panels). The mapper addresses LEDs linearly via `/win` using `S/S2`.
- Disable any existing LED map in WLED. Existing remaps will break index-to-LED matching.

### 2) Map LEDs
1. Open `http://localhost:8000/`.
2. In Controls, set:
   - Host (e.g., `http://10.0.0.50`)
   - LEDs (your strip length)
   - Segment (usually 0)
   - Camera (index for your system)
   - Frames/LED + Min Bright (detection tuning)
   - Prelight/Capture/Postlight delays (stability tuning)
   - Transition (ms) — WLED `TT`, set 0 for instant changes
3. Click **Start Mapping**. The backend will:
   - Turn all LEDs off → light a single LED → off (repeats for each LED)
   - Capture camera frames and store positions in `data/mapping.csv`
4. When done, **Reload CSV** if needed to refresh the UI camera/grid views.

### 3) Shape the grid and export
- Adjust Step/Width/Height as needed.
- Click **Apply** to re-compute the grid and update the preview.
- Click **Generate ledmap.json** to download the LED map formatted for WLED (flattened `map` with `width`/`height`).
- Click **Download 2D gaps** to download a `1/-1` grid mask.

Notes:
- `ledmap.json` and `2d-gaps.json` are not saved on the server; they download in your browser. The raw capture stays in `data/mapping.csv`.
- The UI toggle “Wire LEDs (in order)” draws a red line connecting LEDs 0→N in camera space for quick sanity checks.

## CLI usage
You can run the mapping script directly:
```bash
python scripts/map_leds.py http://wled.local 150 --camera-index 1 --output data/mapping.csv
```
and convert to JSON:
```bash
python scripts/convert_mapping.py --step 15 --include-meta
```

## Troubleshooting
- **ModuleNotFoundError: fastapi** - install dependencies with `pip install -r requirements.txt`.
- **Camera not opened** - verify the camera index and close other apps using it.
- **WLED doesn’t light**
  - Confirm host/IP is reachable and WLED UI opens.
  - Verify a single segment spans the entire strip.
  - Test manual URLs (see above) — ensure `&SS`, `&S/&S2`, `&A`, and `&TT` are present.
  - Increase Prelight/Capture/Postlight delays if your camera needs more time.

## Project structure
- Backend API: `server.py`
- Mapping logic: `scripts/map_leds.py`
- Conversion/trimming logic: `scripts/convert_mapping.py`
- Frontend UI: `ui/led_viewer.html`

Contributions are welcome! Fork, branch, and open a PR.
