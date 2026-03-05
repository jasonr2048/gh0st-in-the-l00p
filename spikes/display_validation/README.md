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

## Method Options
- **VLC** — fast to test but limited programmatic control over scale/crop
- **Python + pygame** — two windows, full control over transform parameters ✓
- **Python + OpenCV** — similar, `cv2.moveWindow()` for monitor placement ✓

Recommendation: go with Python + pygame or OpenCV from the start given the 
adjustability requirement — VLC won't scale well into this.

## Success Criteria
- Both monitors display full-screen portrait video simultaneously
- Scale and crop adjustable via config, no code changes needed
- Correct orientation confirmed
- No significant lag or sync drift between the two

## Next Step
→ Video generation spike (AI-generated face-like visuals, noise/static, portrait format)

