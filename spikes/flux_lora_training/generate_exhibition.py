#!/usr/bin/env python3
"""
FLUX LoRA exhibition generation.

Generates FLUX stills for every style in the selection file, plus 30/70 and
70/30 mashup stills between consecutive styles within each category and at
category boundaries (needed for smooth screen A and screen B videos).

Screen A sequence: stable → ambiguous → glitch → extreme → synthetic
Screen B sequence: extreme → synthetic → stable → ambiguous → glitch

Input:
  /workspace/exhibition_source/selection_v1.txt  (or --selection)
  /workspace/exhibition_source/{category}/{filename}  (source images)

Output:
  /workspace/output/gh0st_exhibition_v1/{category}/{style}.png   (pure stills)
  /workspace/output/gh0st_exhibition_v1/{category}/030A__070B.png  (mashups)
  /workspace/output/gh0st_exhibition_v1/boundaries/{cat_a}_x_{cat_b}/030A__070B.png

Usage:
  python generate_exhibition.py
  python generate_exhibition.py --selection selection_v2.txt --output gh0st_exhibition_v2
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE       = Path("/workspace")
SOURCE_DIR      = WORKSPACE / "exhibition_source"
LORA_PATH       = WORKSPACE / "output" / "gh0st_flux_lora_v2" / "gh0st_flux_lora_v2.safetensors"
MODELS_DIR      = WORKSPACE / "models"
FLUX_MODEL      = "black-forest-labs/FLUX.1-dev"
REDUX_MODEL     = "black-forest-labs/FLUX.1-Redux-dev"

INFERENCE_STEPS = 20
GUIDANCE_SCALE  = 3.5
LORA_SCALE      = 0.90
GEN_SIZE        = 1024
SEED            = 42

MASHUP_ALPHAS   = [0.70, 0.50, 0.30]   # smooth ramp: 70/30 → 50/50 → 30/70

# Category order for each screen — determines which boundary pairs we need
SCREEN_A = ["stable", "ambiguous", "glitch", "extreme", "synthetic"]
SCREEN_B = ["extreme", "synthetic", "stable", "ambiguous", "glitch"]


# ---------------------------------------------------------------------------
# Parse selection file
# ---------------------------------------------------------------------------

def parse_selection(path: Path) -> dict[str, list[Path]]:
    """Return ordered dict: {category: [source_image_path, ...]}."""
    categories: dict[str, list[Path]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("/", 1)
        if len(parts) != 2:
            print(f"  WARNING: skipping malformed line: {line!r}")
            continue
        cat, fname = parts
        img_path = SOURCE_DIR / cat / fname
        if not img_path.exists():
            print(f"  WARNING: source image not found: {img_path}")
            continue
        categories.setdefault(cat, []).append(img_path)
    return categories


def stem(path: Path) -> str:
    """Sanitised filename stem for use in output filenames."""
    s = path.stem
    # Replace spaces and brackets with underscore, collapse runs
    s = re.sub(r"[\s()]+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipelines():
    from diffusers import FluxPriorReduxPipeline, FluxPipeline

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading FluxPriorReduxPipeline...")
    pipe_prior = FluxPriorReduxPipeline.from_pretrained(
        REDUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    ).to("cuda")

    print("Loading FluxPipeline + LoRA...")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    )
    pipe.load_lora_weights(str(LORA_PATH), adapter_name="gh0st")
    pipe.set_adapters(["gh0st"], adapter_weights=[LORA_SCALE])
    pipe.enable_sequential_cpu_offload()
    print("Pipelines ready.")
    return pipe_prior, pipe


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (GEN_SIZE, GEN_SIZE):
        img = img.resize((GEN_SIZE, GEN_SIZE), Image.LANCZOS)
    return img


def encode(pipe_prior, image: Image.Image):
    return pipe_prior(image=image)


def generate(pipe, embeds, pooled, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def blend(prior_a, prior_b, alpha_a: float):
    """Blend two Redux priors: alpha_a * A + (1-alpha_a) * B."""
    alpha_b = 1.0 - alpha_a
    embeds = alpha_a * prior_a.prompt_embeds + alpha_b * prior_b.prompt_embeds
    pooled = alpha_a * prior_a.pooled_prompt_embeds + alpha_b * prior_b.pooled_prompt_embeds
    return embeds, pooled


def mashup_filename(stem_a: str, stem_b: str, alpha_a: float) -> str:
    pct_a = int(round(alpha_a * 100))
    pct_b = 100 - pct_a
    return f"{pct_a:03d}{stem_a}__{pct_b:03d}{stem_b}.png"


# ---------------------------------------------------------------------------
# Count work
# ---------------------------------------------------------------------------

def count_todo(selection: dict[str, list[Path]], out_base: Path) -> int:
    total = 0
    for cat, paths in selection.items():
        cat_dir = out_base / cat
        # pure stills
        for p in paths:
            if not (cat_dir / f"{stem(p)}.png").exists():
                total += 1
        # within-category mashups
        for i in range(len(paths) - 1):
            sa, sb = stem(paths[i]), stem(paths[i + 1])
            for alpha in MASHUP_ALPHAS:
                if not (cat_dir / mashup_filename(sa, sb, alpha)).exists():
                    total += 1

    # boundary mashups
    boundary_pairs = _boundary_pairs(selection)
    for (cat_a, last_a), (cat_b, first_b) in boundary_pairs:
        bdir = out_base / "boundaries" / f"{cat_a}_x_{cat_b}"
        sa, sb = stem(last_a), stem(first_b)
        for alpha in MASHUP_ALPHAS:
            if not (bdir / mashup_filename(sa, sb, alpha)).exists():
                total += 1
    return total


def _boundary_pairs(selection):
    """Return list of ((cat_a, last_path_a), (cat_b, first_path_b)) for all
    unique cross-category adjacent pairs across both screen sequences."""
    seen = set()
    pairs = []
    for seq in (SCREEN_A, SCREEN_B):
        for i in range(len(seq) - 1):
            cat_a, cat_b = seq[i], seq[i + 1]
            key = (cat_a, cat_b)
            if key in seen:
                continue
            if cat_a not in selection or cat_b not in selection:
                continue
            seen.add(key)
            pairs.append(((cat_a, selection[cat_a][-1]), (cat_b, selection[cat_b][0])))
    return pairs


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_all(selection: dict[str, list[Path]], out_base: Path) -> None:
    todo = count_todo(selection, out_base)
    if todo == 0:
        print("All outputs already exist — nothing to do.")
        return

    print(f"\nTotal to generate: {todo} stills")
    pipe_prior, pipe = load_pipelines()

    generated = 0

    # ── Encode all source images upfront ────────────────────────────────────
    print(f"\nEncoding {sum(len(v) for v in selection.values())} source images...")
    priors: dict[str, object] = {}   # stem -> prior
    for cat, paths in selection.items():
        for p in paths:
            s = stem(p)
            if s in priors:
                continue
            print(f"  encoding {cat}/{p.name}")
            priors[s] = encode(pipe_prior, load_image(p))
    del pipe_prior
    torch.cuda.empty_cache()
    print("Encoding done.\n")

    # ── Pure stills + within-category mashups ───────────────────────────────
    for cat, paths in selection.items():
        cat_dir = out_base / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== {cat} ({len(paths)} styles) ===")

        for i, p in enumerate(paths):
            sa = stem(p)

            # Pure still
            pure_out = cat_dir / f"{sa}.png"
            if not pure_out.exists():
                print(f"  [{generated + 1}/{todo}] pure  {sa}")
                generate(pipe, priors[sa].prompt_embeds, priors[sa].pooled_prompt_embeds, pure_out)
                generated += 1
            else:
                print(f"  skip pure {sa}")

            # Mashups with next style in category
            if i < len(paths) - 1:
                sb = stem(paths[i + 1])
                for alpha_a in MASHUP_ALPHAS:
                    mout = cat_dir / mashup_filename(sa, sb, alpha_a)
                    if not mout.exists():
                        print(f"  [{generated + 1}/{todo}] mashup {mashup_filename(sa, sb, alpha_a)}")
                        e, pe = blend(priors[sa], priors[sb], alpha_a)
                        generate(pipe, e, pe, mout)
                        generated += 1
                    else:
                        print(f"  skip mashup {mashup_filename(sa, sb, alpha_a)}")

    # ── Cross-category boundary mashups ─────────────────────────────────────
    print("\n=== boundaries ===")
    for (cat_a, last_a), (cat_b, first_b) in _boundary_pairs(selection):
        sa, sb = stem(last_a), stem(first_b)
        bdir = out_base / "boundaries" / f"{cat_a}_x_{cat_b}"
        bdir.mkdir(parents=True, exist_ok=True)
        for alpha_a in MASHUP_ALPHAS:
            mout = bdir / mashup_filename(sa, sb, alpha_a)
            if not mout.exists():
                print(f"  [{generated + 1}/{todo}] boundary {cat_a}→{cat_b} {mashup_filename(sa, sb, alpha_a)}")
                e, pe = blend(priors[sa], priors[sb], alpha_a)
                generate(pipe, e, pe, mout)
                generated += 1
            else:
                print(f"  skip boundary {mashup_filename(sa, sb, alpha_a)}")

    print(f"\nGenerated {generated} new stills.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(out_base: Path, selection: dict[str, list[Path]]) -> None:
    print("\n=== Summary ===")
    total = 0
    for cat in selection:
        cat_dir = out_base / cat
        n = len(list(cat_dir.glob("*.png"))) if cat_dir.exists() else 0
        print(f"  {cat:12s}: {n} stills")
        total += n
    bdir = out_base / "boundaries"
    nb = sum(len(list(d.glob("*.png"))) for d in bdir.iterdir() if d.is_dir()) if bdir.exists() else 0
    print(f"  boundaries  : {nb} stills")
    print(f"  Total       : {total + nb}")
    print(f"Output at: {out_base}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX exhibition generation")
    parser.add_argument("--selection", default="selection_v1.txt",
                        help="Selection file name (relative to exhibition_source/)")
    parser.add_argument("--output", default="gh0st_exhibition_v1",
                        help="Output directory name under workspace/output/")
    args = parser.parse_args()

    selection_path = SOURCE_DIR / args.selection
    if not selection_path.exists():
        sys.exit(f"ERROR: selection file not found: {selection_path}")

    out_base = WORKSPACE / "output" / args.output
    out_base.mkdir(parents=True, exist_ok=True)

    print("=== FLUX LoRA Exhibition Generation ===")
    print(f"  Selection : {selection_path}")
    print(f"  LoRA      : {LORA_PATH}")
    print(f"  Output    : {out_base}")

    selection = parse_selection(selection_path)
    total_styles = sum(len(v) for v in selection.values())
    print(f"  Styles    : {total_styles} across {len(selection)} categories")
    for cat, paths in selection.items():
        print(f"    {cat}: {len(paths)}")

    generate_all(selection, out_base)
    print_summary(out_base, selection)
    print("\nDONE")
