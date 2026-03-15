# Spike Findings

Outcomes from completed spikes. For full context, method, and running instructions,
see each spike's README.

---

## Display Validation
**Spike:** `spikes/display_validation/`
**Status:** Partially validated — single-screen only, dual-monitor pending hardware

**Findings:**
- OpenCV with two named windows (`WINDOW_NORMAL`) drives simultaneous playback from
  a single process
- Video and window dimensions must be set in config to match the actual monitor
  resolution — this is what ensures the video fills the screen correctly
- Aspect ratio is therefore fixed by the config dimensions (which should match the
  monitor's native ratio)
- Other centering/fill approaches were tried and did not work
- Both screens share a single frame delay derived from screen A's fps — minor
  limitation if the two screens ever need different framerates

**Open questions:**
- Dual-monitor validation still needed on actual hardware