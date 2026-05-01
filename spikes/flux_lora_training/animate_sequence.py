#!/usr/bin/env python3
"""
Runway Gen-4 Turbo — animate sequence stills into 5-second face clips.

Reads the sequence images fetched from the pod:
  spikes/flux_lora_training/output/gh0st_flux_lora_v2/sequence/

For each still, submits an image-to-video task to Runway API and downloads
the resulting clip. Skips stills that already have a clip. Polls until done.

Output: spikes/flux_lora_training/output/gh0st_flux_lora_v2/sequence/clips/

Usage:
  python spikes/flux_lora_training/animate_sequence.py
  python spikes/flux_lora_training/animate_sequence.py --dry-run   # show plan only
  python spikes/flux_lora_training/animate_sequence.py --version v1 # only v1 images
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent
KEY_FILE     = ROOT.parent.parent / ".runway_api_key"          # repo root
SEQUENCE_DIR = ROOT / "output" / "gh0st_flux_lora_v2" / "sequence"
CLIPS_DIR    = SEQUENCE_DIR / "clips"

API_BASE     = "https://api.dev.runwayml.com/v1"
API_VERSION  = "2024-11-06"
MODEL        = "gen4_turbo"
CLIP_DURATION = 5           # seconds; 5 is max for gen4_turbo
PROMPT_TEXT  = (
    "subtle realistic face animation, gentle natural blink, soft micro-expressions, "
    "slight breathing movement, eyes alive, hair barely moves, neutral steady gaze, "
    "no dramatic motion, no camera movement, portrait, black background"
)

POLL_INTERVAL_S = 8      # seconds between status polls
MAX_WAIT_S      = 300    # bail if a single task takes longer than this

# ---------------------------------------------------------------------------
# Sequence definition — must match generate_sequence.py
# ---------------------------------------------------------------------------
TRANSITIONS = [
    ("bead_cage",      "rhinestones"),
    ("crybabyglitch",  "bead_cage"),
    ("red_blackliner", "crybabyglitch"),
]
PURE_STYLES = ["rhinestones", "bead_cage", "crybabyglitch", "red_blackliner"]
ALL_ALPHAS  = [round(v / 100, 2) for v in range(5, 100, 5)]


def load_api_key() -> str:
    if not KEY_FILE.exists():
        sys.exit(f"ERROR: API key not found at {KEY_FILE}")
    return KEY_FILE.read_text().strip()


def headers(key: str) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "X-Runway-Version": API_VERSION,
        "Content-Type": "application/json",
    }


def image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def submit_task(key: str, image_path: Path) -> str:
    """Submit an img2vid task; return task ID."""
    payload = {
        "model": MODEL,
        "promptImage": image_to_data_uri(image_path),
        "promptText": PROMPT_TEXT,
        "duration": CLIP_DURATION,
        "ratio": "768:1344",    # portrait 9:16-ish
    }
    resp = requests.post(f"{API_BASE}/image_to_video", headers=headers(key), json=payload)
    if resp.status_code == 429:
        sys.exit("ERROR: Daily generation limit reached. Try again tomorrow or switch to gen3a_turbo.")
    resp.raise_for_status()
    task_id = resp.json()["id"]
    return task_id


def poll_task(key: str, task_id: str, label: str) -> str | None:
    """Poll until task succeeds; return video URL or None on failure."""
    deadline = time.time() + MAX_WAIT_S
    while time.time() < deadline:
        resp = requests.get(f"{API_BASE}/tasks/{task_id}", headers=headers(key))
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "SUCCEEDED":
            outputs = data.get("output", [])
            if outputs:
                return outputs[0]
            print(f"  [{label}] Task succeeded but no output URL?")
            return None
        if status in ("FAILED", "CANCELLED"):
            print(f"  [{label}] Task {status}: {data.get('failure', '')}")
            return None
        print(f"  [{label}] {status} …")
        time.sleep(POLL_INTERVAL_S)
    print(f"  [{label}] Timed out after {MAX_WAIT_S}s")
    return None


def download_clip(url: str, out_path: Path) -> None:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)


def build_frame_list(version: str | None) -> list[tuple[Path, Path]]:
    """
    Return list of (still_path, clip_path) pairs in sequence order.
    version: 'v1', 'v2', or None (use no-suffix first, then v1).
    """
    frames: list[tuple[Path, Path]] = []

    def _resolve_still(candidates: list[Path]) -> Path | None:
        for c in candidates:
            if c.exists():
                return c
        return None

    def _version_candidates(stem: str, directory: Path) -> list[Path]:
        if version:
            return [directory / f"{stem}_{version}.png"]
        # No version specified: try plain name first, then _v1
        return [directory / f"{stem}.png", directory / f"{stem}_v1.png"]

    pure_dir = SEQUENCE_DIR / "pure"
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Sequence: pure rhinestones, then each transition, then pure endpoint
    # Each transition plays ascending alpha (more set_a → more of the destination)
    styles_in_order = ["rhinestones", "bead_cage", "crybabyglitch", "red_blackliner"]

    # Pure endpoints (rhinestones first, then the other three as segment ends)
    def add_pure(style: str):
        candidates = _version_candidates(style, pure_dir)
        still = _resolve_still(candidates)
        if still:
            clip = CLIPS_DIR / f"pure_{style}.mp4"
            frames.append((still, clip))
        else:
            print(f"  WARNING: pure/{style} not found (tried {candidates})")

    # Transition: low alpha → high alpha (set_b → set_a)
    def add_transition(set_a: str, set_b: str):
        pair_dir = SEQUENCE_DIR / f"{set_a}_x_{set_b}"
        pair_clips = CLIPS_DIR / f"{set_a}_x_{set_b}"
        for alpha in ALL_ALPHAS:
            pct_a = int(round(alpha * 100))
            pct_b = 100 - pct_a
            stem = f"{pct_a:03d}{set_a}__{pct_b:03d}{set_b}"
            candidates = _version_candidates(stem, pair_dir)
            still = _resolve_still(candidates)
            if still:
                clip = pair_clips / f"{stem}.mp4"
                frames.append((still, clip))
            else:
                print(f"  WARNING: {set_a}_x_{set_b}/{stem} not found")

    add_pure("rhinestones")
    add_transition("bead_cage", "rhinestones")
    add_pure("bead_cage")
    add_transition("crybabyglitch", "bead_cage")
    add_pure("crybabyglitch")
    add_transition("red_blackliner", "crybabyglitch")
    add_pure("red_blackliner")

    return frames


def animate(frames: list[tuple[Path, Path]], key: str, dry_run: bool = False) -> None:
    pending = [(s, c) for s, c in frames if not c.exists()]
    done_already = len(frames) - len(pending)
    print(f"\nTotal frames : {len(frames)}")
    print(f"Already done : {done_already}")
    print(f"To animate   : {len(pending)}")

    if dry_run:
        print("\n-- DRY RUN — no API calls made --")
        for s, c in pending[:10]:
            print(f"  {s.name} → {c.name}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    if not pending:
        print("Nothing to do.")
        return

    print(f"\nEstimated cost: {len(pending) * CLIP_DURATION * 5} credits "
          f"({len(pending)} clips × {CLIP_DURATION}s × 5 credits/s)\n")

    for i, (still, clip) in enumerate(pending):
        label = f"{i + 1}/{len(pending)} {still.name}"
        print(f"  [{label}] Submitting…")
        try:
            task_id = submit_task(key, still)
        except requests.HTTPError as e:
            print(f"  [{label}] Submit failed: {e}")
            if e.response is not None and e.response.status_code == 429:
                print("  Daily limit hit — stopping. Re-run tomorrow.")
                break
            continue

        print(f"  [{label}] task={task_id}")
        url = poll_task(key, task_id, label)
        if url:
            download_clip(url, clip)
            print(f"  [{label}] ✓ saved {clip.name}")
        else:
            print(f"  [{label}] ✗ failed")

    done = sum(1 for _, c in frames if c.exists())
    print(f"\nDone: {done}/{len(frames)} clips ready")
    remaining = [s.name for s, c in frames if not c.exists()]
    if remaining:
        print(f"Remaining ({len(remaining)}): {remaining[:5]}{'…' if len(remaining) > 5 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Runway sequence animator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without making API calls")
    parser.add_argument("--version", choices=["v1", "v2"], default=None,
                        help="Which image version to animate (default: auto)")
    args = parser.parse_args()

    key = load_api_key()

    # Quick API check
    resp = requests.get(f"{API_BASE}/organization",
                        headers={"Authorization": f"Bearer {key}",
                                 "X-Runway-Version": API_VERSION})
    resp.raise_for_status()
    balance = resp.json().get("creditBalance", "?")
    print(f"Runway credit balance: {balance}")

    frames = build_frame_list(args.version)
    animate(frames, key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
