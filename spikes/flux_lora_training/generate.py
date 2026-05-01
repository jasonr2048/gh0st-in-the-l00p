#!/usr/bin/env python3
"""
FLUX LoRA v2 — batch generation script
Runs on the RunPod pod (RTX 4090, /workspace layout).

Phase 1 — img2img clean:
  Each training image → FluxImg2ImgPipeline + LoRA, strength ~0.65
  Preserves original style/composition; reinforces identity via LoRA + caption prompt.

Phase 2 — Redux mashups:
  Two style anchors (one representative image per set) → FluxPriorReduxPipeline
  Embeddings blended at specified alphas → FluxPipeline + LoRA
  Produces precise "X% set_A, Y% set_B" face images.

Usage (on pod):
  cd /workspace
  python generate.py              # both phases
  python generate.py --phase 1    # img2img only
  python generate.py --phase 2    # mashups only
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# FLUX.1-dev needs expandable segments to avoid fragmentation OOM on 24GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE       = Path("/workspace")
DATASET_DIR     = WORKSPACE / "dataset" / "lora_training_v2"
LORA_PATH       = WORKSPACE / "output" / "gh0st_flux_lora_v2" / "gh0st_flux_lora_v2.safetensors"
OUTPUT_DIR      = WORKSPACE / "output" / "gh0st_flux_lora_v2" / "generated"
FLUX_MODEL      = "black-forest-labs/FLUX.1-dev"
REDUX_MODEL     = "black-forest-labs/FLUX.1-Redux-dev"

IMG2IMG_STRENGTH = 0.65
INFERENCE_STEPS  = 20     # 20 is sufficient; reduces memory pressure vs 28
GUIDANCE_SCALE   = 3.5
LORA_SCALE       = 0.90   # slightly higher than training default; tune if needed
GEN_SIZE         = 1024
SEED             = 42

# ---------------------------------------------------------------------------
# Mashup pairs
# Format: (set_a, set_b, [alpha values where alpha = weight of set_a])
# ---------------------------------------------------------------------------
MASHUP_PAIRS = [
    # Full ratio sweep — primary "influence" exploration
    ("blue_face",      "rhinestones",    [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]),
    ("black_feathers", "gold_hardware",  [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]),
    ("red_blackliner", "crybabyglitch",  [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]),
    ("neutral",        "tactical",       [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]),
    ("bead_cage",      "rhinestones",    [0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.20, 0.10, 0.05]),
    # Secondary pairs — 3-point snapshot
    ("cyber_pastel",   "blue_face",      [0.70, 0.50, 0.30]),
    ("gold_hardware",  "rhinestones",    [0.70, 0.50, 0.30]),
    ("red_half_1",     "red_blackliner", [0.70, 0.50, 0.30]),
    ("black_crosses",  "crybabyglitch",  [0.70, 0.50, 0.30]),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image_rgb(path: Path, size: int = GEN_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img

def read_caption(image_path: Path) -> str:
    txt = image_path.with_suffix(".txt")
    if txt.exists():
        return txt.read_text().strip()
    return "sks person, portrait, black background"

def get_representative_image(set_name: str) -> Path:
    """Return the middle image from a set as a stable style anchor."""
    set_dir = DATASET_DIR / set_name
    images = sorted(
        p for p in set_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not images:
        raise FileNotFoundError(f"No images found in {set_dir}")
    return images[len(images) // 2]

# ---------------------------------------------------------------------------
# Phase 1: img2img clean versions
# ---------------------------------------------------------------------------
def phase1_img2img():
    from diffusers import FluxImg2ImgPipeline

    print("\n[Phase 1] Loading FluxImg2ImgPipeline + LoRA...")
    pipe = FluxImg2ImgPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(str(LORA_PATH), adapter_name="gh0st")
    pipe.set_adapters(["gh0st"], adapter_weights=[LORA_SCALE])
    pipe.enable_sequential_cpu_offload()  # more aggressive offload — needed on 24GB with LoRA
    print("  Pipeline ready.")

    clean_dir = OUTPUT_DIR / "clean"
    sets = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())

    total_done = 0
    for set_dir in sets:
        images = sorted(
            p for p in set_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not images:
            continue
        out_dir = clean_dir / set_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Set: {set_dir.name} ({len(images)} images)")

        for img_path in images:
            out_path = out_dir / img_path.with_suffix(".png").name
            if out_path.exists():
                print(f"    skip {img_path.name}")
                continue
            caption = read_caption(img_path)
            source  = load_image_rgb(img_path)
            generator = torch.Generator("cuda").manual_seed(SEED)
            result = pipe(
                prompt=caption,
                image=source,
                strength=IMG2IMG_STRENGTH,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
            result.save(out_path)
            total_done += 1
            print(f"    saved {out_path.name}")

    del pipe
    torch.cuda.empty_cache()
    print(f"\n[Phase 1] Done — {total_done} images generated.")

# ---------------------------------------------------------------------------
# Phase 2: Redux mashups
# ---------------------------------------------------------------------------
def phase2_mashups():
    from diffusers import FluxPriorReduxPipeline, FluxPipeline

    print("\n[Phase 2] Loading FluxPriorReduxPipeline (Redux encoder)...")
    pipe_prior = FluxPriorReduxPipeline.from_pretrained(
        REDUX_MODEL, torch_dtype=torch.bfloat16
    ).to("cuda")

    print("[Phase 2] Loading FluxPipeline + LoRA...")
    pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(str(LORA_PATH), adapter_name="gh0st")
    pipe.set_adapters(["gh0st"], adapter_weights=[LORA_SCALE])
    pipe.enable_sequential_cpu_offload()  # more aggressive offload — needed on 24GB with LoRA
    print("  Pipelines ready.")

    mashup_dir = OUTPUT_DIR / "mashups"
    total_done = 0

    for set_a, set_b, alphas in MASHUP_PAIRS:
        pair_label = f"{set_a}_x_{set_b}"
        pair_dir   = mashup_dir / pair_label
        pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Pair: {set_a} × {set_b}")

        # Encode anchor images once per pair
        try:
            anchor_a = load_image_rgb(get_representative_image(set_a))
            anchor_b = load_image_rgb(get_representative_image(set_b))
        except FileNotFoundError as e:
            print(f"    ERROR: {e} — skipping pair")
            continue

        print(f"    Encoding {set_a}...")
        prior_a = pipe_prior(image=anchor_a)
        print(f"    Encoding {set_b}...")
        prior_b = pipe_prior(image=anchor_b)

        for alpha in alphas:
            pct_a = int(round(alpha * 100))
            pct_b = 100 - pct_a
            label    = f"{pct_a:03d}{set_a}__{pct_b:03d}{set_b}"
            out_path = pair_dir / f"{label}.png"
            if out_path.exists():
                print(f"    skip {label}")
                continue

            # Blend embeddings
            embeds = (
                alpha * prior_a.prompt_embeds
                + (1 - alpha) * prior_b.prompt_embeds
            )
            pooled = (
                alpha * prior_a.pooled_prompt_embeds
                + (1 - alpha) * prior_b.pooled_prompt_embeds
            )

            generator = torch.Generator("cuda").manual_seed(SEED)
            result = pipe(
                prompt_embeds=embeds,
                pooled_prompt_embeds=pooled,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=GEN_SIZE,
                width=GEN_SIZE,
                generator=generator,
            ).images[0]
            result.save(out_path)
            total_done += 1
            print(f"    saved {label}.png")

    del pipe_prior, pipe
    torch.cuda.empty_cache()
    print(f"\n[Phase 2] Done — {total_done} mashups generated.")

# ---------------------------------------------------------------------------
# Grid helpers — make results easy to review at a glance
# ---------------------------------------------------------------------------
LABEL_H   = 36   # px height for text label bar below each image
THUMB_W   = 256  # thumbnail width for grid cells
THUMB_H   = 256  # thumbnail height

def _thumb(img: Image.Image) -> Image.Image:
    return img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

def _label_bar(text: str, width: int, height: int = LABEL_H,
               bg: tuple = (20, 20, 20), fg: tuple = (220, 220, 220)) -> Image.Image:
    bar = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = max(4, (width - (bbox[2] - bbox[0])) // 2)
    y = max(2, (height - (bbox[3] - bbox[1])) // 2)
    draw.text((x, y), text, font=font, fill=fg)
    return bar

def _cell(img: Image.Image, label: str) -> Image.Image:
    """Thumbnail + label bar stacked vertically."""
    t = _thumb(img)
    bar = _label_bar(label, THUMB_W)
    cell = Image.new("RGB", (THUMB_W, THUMB_H + LABEL_H))
    cell.paste(t, (0, 0))
    cell.paste(bar, (0, THUMB_H))
    return cell

def make_mashup_grid(pair_dir: Path, set_a: str, set_b: str):
    """
    For one mashup pair: produce a single horizontal strip showing
    all ratio steps from 100% set_a → 100% set_b, left to right.
    Saved as  pair_dir / _grid.jpg
    """
    images_paths = sorted(pair_dir.glob("*.png"))
    if not images_paths:
        return
    cells = []
    for p in images_paths:
        # filename: e.g.  095blue_face__005rhinestones.png
        # derive a short label like "95 / 5"
        stem = p.stem
        parts = stem.split("__")
        if len(parts) == 2:
            pct_a = parts[0][:3].lstrip("0") or "0"
            pct_b = parts[1][:3].lstrip("0") or "0"
            label = f"{pct_a}% {set_a}\n{pct_b}% {set_b}"
        else:
            label = stem
        cells.append(_cell(Image.open(p).convert("RGB"), label))

    # Assemble horizontal strip
    gap = 4
    total_w = len(cells) * THUMB_W + (len(cells) - 1) * gap
    total_h = THUMB_H + LABEL_H
    strip = Image.new("RGB", (total_w, total_h), (8, 8, 8))
    for i, cell in enumerate(cells):
        strip.paste(cell, (i * (THUMB_W + gap), 0))

    # Add a title bar at the top
    title_bar = _label_bar(
        f"{set_a}  ←  ratio  →  {set_b}",
        total_w, height=28, bg=(40, 40, 40), fg=(255, 200, 100)
    )
    final = Image.new("RGB", (total_w, 28 + total_h), (8, 8, 8))
    final.paste(title_bar, (0, 0))
    final.paste(strip, (0, 28))
    final.save(pair_dir / "_grid.jpg", quality=92)
    print(f"  grid saved: {pair_dir.name}/_grid.jpg  ({len(cells)} cells)")

def make_clean_grid(set_name: str, originals: list[Path], generated_dir: Path):
    """
    For one set: side-by-side original | generated for every image.
    Saved as  generated_dir / _grid.jpg
    """
    cells = []
    for orig_path in originals:
        gen_path = generated_dir / orig_path.with_suffix(".png").name
        if not gen_path.exists():
            continue
        orig_img = Image.open(orig_path).convert("RGB")
        gen_img  = Image.open(gen_path).convert("RGB")
        left  = _cell(orig_img, "original")
        right = _cell(gen_img,  "generated")
        # Pair cell: two thumbs side by side with a 2px divider
        pair_w = THUMB_W * 2 + 2
        pair = Image.new("RGB", (pair_w, THUMB_H + LABEL_H), (50, 50, 50))
        pair.paste(left,  (0, 0))
        pair.paste(right, (THUMB_W + 2, 0))
        cells.append(pair)

    if not cells:
        return
    gap = 6
    cols = min(len(cells), 4)   # up to 4 pairs per row
    rows = (len(cells) + cols - 1) // cols
    cell_w = THUMB_W * 2 + 2
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows * (THUMB_H + LABEL_H) + (rows - 1) * gap + 30
    grid = Image.new("RGB", (total_w, total_h), (8, 8, 8))
    title_bar = _label_bar(f"{set_name} — original vs generated", total_w, height=28,
                           bg=(40, 40, 40), fg=(255, 200, 100))
    grid.paste(title_bar, (0, 0))
    y_off = 30
    for i, cell in enumerate(cells):
        col = i % cols
        row = i // cols
        x = col * (cell_w + gap)
        y = y_off + row * (THUMB_H + LABEL_H + gap)
        grid.paste(cell, (x, y))
    grid.save(generated_dir / "_grid.jpg", quality=92)
    print(f"  grid saved: {set_name}/_grid.jpg  ({len(cells)} pairs)")

def phase3_grids():
    """Build review grids for both clean and mashup outputs."""
    print("\n[Phase 3] Building review grids...")

    # --- mashup grids ---
    mashup_root = OUTPUT_DIR / "mashups"
    if mashup_root.exists():
        for set_a, set_b, _ in MASHUP_PAIRS:
            pair_dir = mashup_root / f"{set_a}_x_{set_b}"
            if pair_dir.exists():
                make_mashup_grid(pair_dir, set_a, set_b)

    # --- clean grids ---
    clean_root = OUTPUT_DIR / "clean"
    if clean_root.exists():
        for set_dir in sorted(d for d in DATASET_DIR.iterdir() if d.is_dir()):
            originals = sorted(
                p for p in set_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            gen_dir = clean_root / set_dir.name
            if gen_dir.exists() and originals:
                make_clean_grid(set_dir.name, originals, gen_dir)

    print("[Phase 3] Grids done.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="1=img2img  2=mashups  3=grids only  (default: all three)")
    args = parser.parse_args()

    print("=== FLUX LoRA v2 Generation ===")
    print(f"  LoRA:    {LORA_PATH}")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in (None, 1):
        phase1_img2img()
    if args.phase in (None, 2):
        phase2_mashups()
    if args.phase in (None, 3):
        phase3_grids()

    # Summary
    clean   = list((OUTPUT_DIR / "clean").rglob("*.png"))   if (OUTPUT_DIR / "clean").exists()   else []
    mashups = list((OUTPUT_DIR / "mashups").rglob("*.png")) if (OUTPUT_DIR / "mashups").exists() else []
    grids   = list(OUTPUT_DIR.rglob("_grid.jpg"))
    print(f"\n=== Summary ===")
    print(f"  Clean images : {len(clean)}")
    print(f"  Mashup images: {len(mashups)}")
    print(f"  Review grids : {len(grids)}")
    print(f"  Total        : {len(clean) + len(mashups)}")
    print(f"\nOutput at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
