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
**Video generation for A to review.** Goal: optical flow interpolation video
showing A's face morphing through multiple makeup styles, portrait format (1080x1920).

### Active work
- `notebooks/optical_flow_interpolation.ipynb` — main notebook
  - Scaling/portrait fix applied (512x512 → centred on 1080x1920 canvas)
  - Known issue: face position/orientation varies between images → jarring transitions
  - Next: keypoint-based image ordering to minimise face position jumps between frames
- `tools/optimise_sequence.py` — TO BE WRITTEN
  - Use YOLOv8 pose keypoints (nose + eyes) to order images within and across sets
  - Minimise total keypoint distance across the sequence
  - Output: `data/sequence.json` consumed by interpolation notebook

### Parallel explorations (not started)
- Diffusion-based stylisation with identity preservation (InstantID / IP-Adapter)
- Face reenactment tools (Runway, Hedra, Kling)
- StyleGAN2-ADA fine-tuning on A's dataset (longer term)
- Video post-processing: text overlay, sound, visual scan effect (separate spike)
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
