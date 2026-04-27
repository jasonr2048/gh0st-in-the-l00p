# CLAUDE.md

This file provides context for Claude Code sessions on this project.
For broader planning and architecture discussions, see the claude.ai Project.

## Project Overview
An art installation called "Gh0st in the L00p" featuring two screens displaying
AI-generated face-like visuals of A's face in various makeup styles, creating
a feedback loop of mutual perception. Two entities that exist in parallel,
aware of and responding to each other.

**Deadline:** May 7-10, 2026, Rostock

## Team
- Jason — sole technical person (Python, ML, CV, hardware)
- A — artistic collaborator, visual datasets, concept

## Key Constraints
- Installation can "cheat" — pre-recorded, selected, looped content is fine.
  **Reliability > real-time performance.**
- Output must be recognisably A's face, abstracted to varying degrees.
  Not generic synthetic faces.
- Two separate portrait screens (42-48" OLED), each playing a looping video
- Hardware sync handled by venue (two Arduinos with looper function over LAN)
- Life-size+ face display, scale/crop tunable via config at runtime
- Text overlay/below video and sound: handled by A
- Visual scan effect to be added to videos (separate spike)

## Stack
- Python (primary language)
- YOLOv8 pose estimation for head detection (not face detection — makeup defeats face detectors)
- rembg[cpu] for background removal
- OpenCV for optical flow interpolation
- Google Colab for heavy compute (Intel Mac x86_64 — avoid local PyTorch installs)

## Repo Structure
```
notebooks/       # Colab notebooks
  dataset_prep.ipynb
  optical_flow_interpolation.ipynb
  stylegan_ffhq_projection.ipynb  # dead end, kept for reference
  background_removal.ipynb
docs/            # planning docs, decisions log
tools/
  prepare_dataset.py   # YOLOv8 pose-based head crop + resize pipeline
  requirements.txt     # Colab deps (not local)
spikes/          # isolated technical spikes
data/            # dataset tags and sequence files (images live on Drive)
CLAUDE.md        # this file
```

## Dataset Structure (Google Drive)
```
dataset/
  raw/
    ready/       # 39 style subfolders — source images for pipeline
    tricky/      # bg/lighting issues: black_feathers, black_mesh, camo_studio,
                 #   nat_indian, red_disks, smiley_holo
    concealed/   # face hidden — artistic reference only
    neutral/     # plain face shots, useful as interpolation anchor
    rejected/    # empty
  prepared/      # 39 subfolders, 512x512 square crops from raw/ready/
  bg_removed/    # camo_studio, nat_indian, smiley_holo — bg removed, not yet prepped
```

## Current Phase
**Working pipeline.** Full workflow is functional end-to-end:
Colab pipeline → morph video → local overlay export → two exhibition MP4s.

### What's working
- `notebooks/pipeline.ipynb` — full Colab pipeline:
  bg removal (rembg, corner-check skip) → crop/resize (prepare_dataset_v2.py) →
  sequence optimisation (2-opt, YOLOv8 keypoints) → manual review grid →
  video generation (streaming optical flow, no OOM)
- `app.py` — local CLI: `python app.py --video data/<id>.mp4 --export`
  produces `exports/exhibition/<id>_<ts>/screen_A.mp4` + `screen_B.mp4`
- Sidecar JSON written by Colab, read by local runtime for duration/experiment ID
- `fonts/CourierPrime-Regular.ttf` bundled for cross-machine font consistency

### Known open items
- Visual quality of morph video still being tuned (timing, set selection, bg removal results)
- Diffusion-based stylisation with identity preservation (InstantID / IP-Adapter) — not started
- Face reenactment tools (Runway, Hedra, Kling) — not started
- StyleGAN2-ADA fine-tuning on A's dataset — longer term
- Sound — handled by A
- Two-entity responsiveness: entity B gradually mirrors entity A's current style set

## Key Decisions
See `docs/decisions_log.md` for full log. Summary:
- FFHQ pretrained StyleGAN projection doesn't work — A's face too far outside distribution
- YOLOv8 pose estimation is reliable head detector when makeup defeats face detectors
- Optical flow interpolation chosen for quick prototype (approach B)
- bg_removed sets need to go through prepare_dataset.py before use
- `--scale 2.5` works better than 3.0 for head crop

## Conventions
- Config values never hardcoded — always variables or config files
- Walrus operator where appropriate
- `--overwrite` flag for reprocessing; default skips existing files
- Keep spikes self-contained — code and doc together in spike subfolder
- Notebooks: clear outputs before committing

## Cowork operating principle
**Cowork sessions must run experiments autonomously.** Jason's time is the
bottleneck. Default to scripted/headless/CLI workflows over browser UIs. When
remote infra (RunPod, Colab, etc.) is in play, set up SSH or API access
upfront so the agent can drive the whole experiment end-to-end. Ask Jason
only for things that require his account credentials or physical action
(payment, OAuth, pasting a public key, accepting ToS). Report back with
results, not step-by-step checkpoints that need his input.
