# Gh0st in the L00p

An art installation featuring two portrait OLED screens displaying AI-generated
visuals of A's face morphing through multiple makeup styles. Two entities exist
in parallel, aware of and responding to each other.

**Deadline:** May 7–10, 2026, Rostock

---

## Current deliverables

### 1. Morph video (`notebooks/pipeline.ipynb`)

Run on Google Colab. Produces a portrait 1080×1920 video of A's face flowing
continuously through selected makeup style sets.

**Pipeline:**
1. **Bg removal** — rembg strips backgrounds; images with already-black corners are copied as-is
2. **Dataset prep** — YOLOv8 face + pose crop, centred on mouth, resized to 512×512
3. **Sequence optimisation** — 2-opt ordering by nose/eye keypoints to minimise face-position jumps
4. **Manual review** — per-set grid to reorder images by hand if needed
5. **Video generation** — optical flow (Farneback) morphs, streamed directly to MP4

Output: `<experiment_id>.mp4` + `<experiment_id>.json` (sidecar) saved to Google Drive.

### 2. Exhibition overlay (`app.py`)

Run locally. Takes the morph video as background and composites A's scan line +
terminal text overlay on top, producing two portrait MP4s — one per screen.

```bash
# Export both screens
python app.py --video data/<experiment_id>.mp4 --export

# Preview in windows
python app.py --video data/<experiment_id>.mp4

# Override duration
python app.py --video data/<experiment_id>.mp4 --export --duration 180
```

Output: `exports/exhibition/<experiment_id>_<timestamp>/screen_A.mp4` and `screen_B.mp4`

The overlay reads the sidecar JSON automatically if it's in the same folder as the video.

---

## Full workflow

```
Colab: run pipeline.ipynb
  └─ outputs/<experiment_id>.mp4
  └─ outputs/<experiment_id>.json

Download both files to data/

Local: python app.py --video data/<experiment_id>.mp4 --export
  └─ exports/exhibition/<experiment_id>_<ts>/screen_A.mp4
  └─ exports/exhibition/<experiment_id>_<ts>/screen_B.mp4
```

---

## Repo structure

```
notebooks/
  pipeline.ipynb                    # Main: full pipeline on Colab
  manual_sequence_annotation.ipynb  # Per-set manual ordering override
  optical_flow_interpolation.ipynb  # Standalone interpolation (legacy)
  background_removal.ipynb          # Standalone bg removal (legacy)
  dataset_prep.ipynb                # Standalone prep (legacy)
  stylegan_ffhq_projection.ipynb    # Dead end, kept for reference

tools/
  prepare_dataset_v2.py   # YOLOv8 face + pose crop pipeline (current)
  prepare_dataset.py      # v1 pose-only, kept for reference
  requirements.txt        # Colab pip deps

spikes/
  optical_flow_interpolation/
    optimise_sequence.py  # Keypoint-based 2-opt sequence ordering
  display_validation/     # OpenCV dual-screen validation

exhibition/               # Overlay system (A's work + integration)
  runtime.py              # Still-image backed runtime (original)
  video_runtime.py        # Video-backed runtime (current)
  models.py               # ExhibitionFrameState, VideoFrameState
  text_payload.json       # Text pools for terminal overlay

render/
  exhibition_renderer.py  # Scan line, tinting, text rendering
  dual_screen_renderer.py # OpenCV window management
  utils.py                # fit_to_window helper

fonts/
  CourierPrime-Regular.ttf  # Bundled font for terminal overlay

data/
  dataset_tags.md             # All sets with notes (gitignored except tagged files)
  dataset_tags_exhibition.md  # Exhibition categories per set

docs/
  decisions_log.md
  spike_findings.md
  exhibition_notes.md

app.py      # CLI entry point
config.py   # All tuneable parameters
CLAUDE.md   # Context for Claude Code sessions
```

---

## Stack

- Python 3.12
- OpenCV — optical flow, video I/O, overlay rendering
- YOLOv8 pose + face models — head detection and crop centring
- rembg — background removal
- imageio — video writing on Colab
- Google Colab — heavy compute (Intel Mac x86_64, no local PyTorch)

---

## Dataset (Google Drive)

```
dataset/
  raw/
    ready/     # 39 style subfolders — source for pipeline
    tricky/    # Sets with bg/lighting issues (list in bg_removal_sets)
    neutral/   # Plain face shots, useful as interpolation anchor
    concealed/ # Face hidden — artistic reference only
  bg_removed/  # rembg output, one subfolder per set
  prepared_*/  # 512×512 crops, folder name encodes prep params
```

Images live on Google Drive, not in this repo.

---

## Key config values

`config.py` — overlay system:

| Parameter | Default | Notes |
|---|---|---|
| `scan_cycle_seconds` | 12.0 | Scan line top→bottom duration |
| `proof_duration_seconds` | 72.0 | Overlay export duration fallback |
| `clean_presentation` | True | Hides debug border |

`notebooks/pipeline.ipynb` CONFIG cell — video generation:

| Parameter | Default | Notes |
|---|---|---|
| `hold_frames` | 24 | 1 s hold per image at 24 fps |
| `morph_frames` | 72 | 3 s morph within a set |
| `between_frames` | 96 | 4 s morph between sets |
