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
