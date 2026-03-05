# CLAUDE.md

This file provides context for Claude Code sessions on this project.
For broader planning and architecture discussions, see the claude.ai Project.

## Project Overview
An art installation featuring two screens displaying AI-generated face-like visuals,
each "observing" the other via webcam, creating a feedback loop of mutual perception.
Two AI entities that diverge over time through differing initial states, processing
delays, adaptive feedback parameters, and stochastic noise.

**Deadline:** May 7-10, 2026, Rostock

## Team
- Jason — sole technical person (Python, ML, CV, hardware)
- A — artistic collaborator, visual datasets, concept

## Key Constraint
The installation can "cheat" — pre-recorded, selected, and looped content is
acceptable. **Reliability > real-time performance.**

## Hardware Target
- Two monitors, ideally 42" OLED up to 48", vertical (portrait) orientation
- Single machine driving both displays
- Life-size+ face display, scale/crop tunable via config at runtime

## Stack
- Python (primary language)
- TBC: pygame or OpenCV for display loop
- TBC: CV/ML libraries for face processing

## Repo Structure
```
spikes/          # isolated technical spikes, each with README + code
docs/            # planning docs, decisions log
CLAUDE.md        # this file
```

## Current Phase
**Display validation spike** — see `spikes/display_validation/README.md`

## Decisions Log
See `docs/decisions_log.md`

## Conventions
- Config values (scale, crop, offsets) always in a config file, never hardcoded
- Keep spikes self-contained — code and doc together in spike subfolder
