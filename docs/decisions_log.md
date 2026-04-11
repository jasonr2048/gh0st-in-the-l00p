# Decisions Log

## 2026-03-05

### Single machine, dual display
**Decision:** Run both screens from one machine rather than two separate devices.
**Rationale:** Eliminates inter-device communication, latency, and sync complexity entirely.

### Pre-recorded output acceptable
**Decision:** Installation can use pre-recorded, selected, and looped content.
**Rationale:** Prioritises reliability over real-time processing. Allows use of heavier
models and offline tuning of outputs.

### Python as primary language
**Decision:** Python throughout.
**Rationale:** Jason's strongest language; strong library support for CV, ML, hardware, audio.

### Repo structure
**Decision:** Private GitHub repo, monorepo.
**Structure:** `spikes/` for isolated technical work, `docs/` for planning docs, `CLAUDE.md` in root.
**Rationale:** Spikes self-contained with code + doc together. Avoids premature structure.

### Planning docs in repo
**Decision:** Planning docs live in repo as markdown (`docs/`), synced locally and shared
to Drive for A's visibility. Single source of truth, no duplication.

### Scale/crop tunable at runtime
**Decision:** Face display scale, crop, and position always driven by a config file,
never hardcoded.
**Rationale:** On-site tuning needed once installed in actual space with actual monitors.

## 2026-03-30

### GAN inversion with FFHQ pretrained model — not viable
**Finding:** Direct projection of A's images into FFHQ latent space produces
unrecognisable results. A's face is too far outside FFHQ's training distribution
for meaningful inversion.
**Decision:** Deprioritise FFHQ projection. Fine-tuning StyleGAN2-ADA on A's
dataset remains the correct long-term path but is not this week's priority.

### Pivoting to three parallel approaches for visual prototype
**Decision:** Explore in parallel:
- B: Pixel-space interpolation between prepared images (quick win, recognisably her face)
- C: Diffusion-based stylisation with identity preservation (InstantID / IP-Adapter)
- D: Face reenactment tools (Runway, Hedra, Kling) — try online tools first
**Goal:** Something A can react to visually by Sunday April 5th.

### Output must remain recognisably A's face
**Decision:** Confirmed with Jason — output should be distinctly her face,
abstracted to varying degrees, not generic synthetic faces.
This rules out pure FFHQ generation as a primary output.

## 2026-04-11

### Dataset structure finalised
**Decision:** Dataset organised into `raw/ready/`, `raw/tricky/`, `raw/concealed/`,
`raw/neutral/`, `raw/rejected/`. Prepared images (512x512 square crops) in `prepared/`.
Background-removed versions of tricky sets in `bg_removed/` (not yet run through prep pipeline).
**Note:** `--scale 2.5` preferred over 3.0 for head crop in prepare_dataset.py.

### Keypoint-based sequence optimisation
**Decision:** Use YOLOv8 pose keypoints (nose + eyes) to order images within and
across sets to minimise face position jumps between frames. Sets stay grouped;
within each set and at set boundaries, ordering is optimised for smoothness.
**Implementation:** `tools/optimise_sequence.py` → `data/sequence.json`, consumed
by `notebooks/optical_flow_interpolation.ipynb`.

### Claude Desktop + filesystem MCP
**Decision:** Claude (claude.ai) now has read/write access to the local repo and
Google Drive folder via filesystem connector. Claude Code (Code tab) used for
running scripts. Chat tab for planning/architecture. Cowork tab for file tasks.

### Portrait scaling fix
**Decision:** 512x512 prepared images centred on 1080x1920 black canvas (not stretched).
Face fills full width (1080px), black bars top and bottom.

### Hardware delivery
**Decision:** Two separate looping videos, one per screen. Venue handles sync
via two Arduinos with looper function connected over LAN. No software sync needed.

### Text, sound, scan effect
**Decision:** Text overlay/below video and sound handled by A.
Visual scan effect to be added to videos — separate spike, not yet started.

### Two-entity concept clarification
**Decision:** The two entities don't diverge or converge — they exist in parallel,
aware of and responding to each other. The installation loops, so there's no
endpoint. Entity B could be scripted to respond to A's current style set
(e.g. migrate toward it after a while). To be explored once base video is working.
