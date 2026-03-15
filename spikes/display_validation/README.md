# Display Validation Spike

## Goal
Validate that two videos can be played simultaneously, full-screen, on two separate 
vertical monitors from a single machine, with adjustable scale and crop.

## Hardware Context
- Two monitors, ideally 42" OLED, up to 48", vertical orientation
- Single machine (Mini PC TBC), dual HDMI output
- Target: slightly larger than life-size face, tunable on-site

## Life-Size Reference
- Human face: ~20-22cm wide, 25-28cm tall
- Target display size: slightly above this, exact value to be tuned on-site
- At 42" portrait: screen is ~107cm wide, 60cm tall at standard 16:9 rotated
- A life-size face occupies roughly 20-25% of screen width — so target ~25-35%

## Adjustability Requirement
Scale and crop must be tunable without code changes, via a config file read at startup:
- `scale` — zoom factor relative to life-size reference
- `crop_x`, `crop_y` — crop offsets
- `offset_x`, `offset_y` — position on screen

## Approach
1. Source or generate two portrait-oriented test videos
2. Play both simultaneously, one per monitor, full-screen, with config applied
3. Validate scale, crop, orientation and sync

## Method
Python + OpenCV. Videos are pre-generated at the target monitor resolution and played
back by the player. Generation and playback are independent steps.

## Running

All commands from repo root.

**1. Edit config** (`spikes/display_validation/config.toml`) — set `video_width` and
`video_height` to match the target monitor resolution before generating.

**2. Generate test videos** (re-run whenever config dimensions change):

```sh
uv run spikes/display_validation/generate_test_videos.py
```

**3. Run the player:**

```sh
uv run spikes/display_validation/player.py
```

Two windows open ("SCREEN A" and "SCREEN B"). Videos loop automatically.
Press `Q` (with an OpenCV window focused) or `Ctrl+C` in the terminal to quit.

**On a laptop:** both windows appear on the same screen — fine for a basic sanity check.

**For dual-monitor validation:** drag each window to its target monitor and fullscreen
via the OS (macOS: `Ctrl+Cmd+F`). Set `window_width`/`window_height` in config to
the monitor resolution so the window opens at the right size.

## Success Criteria
- Both monitors display full-screen portrait video simultaneously
- Scale and crop adjustable via config, no code changes needed
- Correct orientation confirmed
- No significant lag or sync drift between the two

## Outcome
**Status:** Partially validated — single-screen only, dual-monitor pending hardware

- Both windows play simultaneously and loop correctly
- Video and window dimensions must match the actual monitor resolution in config —
  this is what makes the video fill the screen correctly; other approaches did not work
- Aspect ratio is fixed by the config dimensions (which should match the monitor's
  native ratio); not adjustable at runtime
- See `docs/spike_findings.md` for the summary

## Next Step
→ Video generation spike (AI-generated face-like visuals, noise/static, portrait format)

