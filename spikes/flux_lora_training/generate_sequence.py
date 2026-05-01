#!/usr/bin/env python3
"""
FLUX LoRA v2 — sequence gap-fill generation
Runs on RunPod pod (/workspace layout).

Generates every 5% ratio step (0.05 → 0.95, 19 steps) for 3 style transitions
that form the sequence video:
  rhinestones → bead_cage        (transition A)
  bead_cage   → crybabyglitch    (transition B — new pair)
  crybabyglitch → red_blackliner (transition C)

Naming convention: alpha = fraction of set_a in blended embedding.
  alpha=0.95 → file 095{set_a}__005{set_b}.png  (mostly set_a)
  alpha=0.05 → file 005{set_a}__095{set_b}.png  (mostly set_b)

For the sequence video, each transition plays from low→high alpha
(i.e. style_b → style_a), so the full sequence reads:
  pure rhinestones
  → [005bead_cage__095rhinestones ... 095bead_cage__005rhinestones]
  → pure bead_cage
  → [005crybabyglitch__095bead_cage ... 095crybabyglitch__005bead_cage]
  → pure crybabyglitch
  → [005red_blackliner__095crybabyglitch ... 095red_blackliner__005crybabyglitch]
  → pure red_blackliner

Output: /workspace/output/gh0st_flux_lora_v2/sequence/{set_a}_x_{set_b}/
Pure-style reference frames are written to:
  /workspace/output/gh0st_flux_lora_v2/sequence/pure/{set_name}.png

Usage:
  python generate_sequence.py              # all transitions, 1 version each
  python generate_sequence.py --versions 2 # generate 2 seeds; v1 + v2 suffixes
"""

import argparse
import os
import sys
import torch
from pathlib import Path
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE    = Path("/workspace")
DATASET_DIR  = WORKSPACE / "dataset" / "lora_training_v2"
LORA_PATH    = WORKSPACE / "output" / "gh0st_flux_lora_v2" / "gh0st_flux_lora_v2.safetensors"
OUTPUT_DIR   = WORKSPACE / "output" / "gh0st_flux_lora_v2" / "sequence"
MODELS_DIR   = WORKSPACE / "models"   # persistent cache — survives pod restarts
FLUX_MODEL   = "black-forest-labs/FLUX.1-dev"
REDUX_MODEL  = "black-forest-labs/FLUX.1-Redux-dev"

INFERENCE_STEPS = 20
GUIDANCE_SCALE  = 3.5
LORA_SCALE      = 0.90
GEN_SIZE        = 1024
SEEDS           = [42, 123]   # seed per version; --versions controls how many to use

# All ratios at every 5% step (pure endpoints handled separately as reference frames)
ALL_ALPHAS = [round(v / 100, 2) for v in range(5, 100, 5)]  # 0.05 … 0.95

# Sequence transitions: (set_a, set_b)
# The video goes set_b → set_a for each, i.e. ascending alpha order.
TRANSITIONS = [
    ("bead_cage",      "rhinestones"),    # transition A: rhinestones → bead_cage
    ("crybabyglitch",  "bead_cage"),      # transition B: bead_cage → crybabyglitch
    ("red_blackliner", "crybabyglitch"),  # transition C: crybabyglitch → red_blackliner
]

# Pure styles needed as anchor / reference frames for the video
PURE_STYLES = ["rhinestones", "bead_cage", "crybabyglitch", "red_blackliner"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image_rgb(path: Path, size: int = GEN_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def get_representative_image(set_name: str) -> Path:
    """Return the middle image for a style.

    Supports two dataset layouts:
    - Subdirectory:  DATASET_DIR/{set_name}/{set_name}_01.png  (prepared dataset)
    - Flat:          DATASET_DIR/{set_name}_01.png             (lora_training_v2)
    """
    set_dir = DATASET_DIR / set_name
    if set_dir.is_dir():
        images = sorted(
            p for p in set_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
    else:
        # Flat dataset: files named {set_name}_01.png etc.
        images = sorted(
            p for p in DATASET_DIR.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            and p.stem.startswith(set_name + "_")
        )
    if not images:
        raise FileNotFoundError(
            f"No images for style '{set_name}' in {DATASET_DIR}"
        )
    return images[len(images) // 2]


def version_suffix(v: int, total: int) -> str:
    """Return '_v1', '_v2' etc. when generating multiple versions; '' for single."""
    return f"_v{v + 1}" if total > 1 else ""


# ---------------------------------------------------------------------------
# Phase: pure reference frames (Redux single-style, no LoRA needed)
# ---------------------------------------------------------------------------
def generate_pure_frames(versions: int = 1):
    """Generate pure single-style reference frames for the sequence endpoints."""
    from diffusers import FluxPriorReduxPipeline, FluxPipeline

    pure_dir = OUTPUT_DIR / "pure"
    pure_dir.mkdir(parents=True, exist_ok=True)

    # Check which pure styles still need generating
    needed = []
    for style in PURE_STYLES:
        for v in range(versions):
            suffix = version_suffix(v, versions)
            out = pure_dir / f"{style}{suffix}.png"
            if not out.exists():
                needed.append((style, v))

    if not needed:
        print("[Pure] All pure frames already exist — skipping.")
        return

    print(f"\n[Pure] Loading pipelines for {len(needed)} pure frame(s)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipe_prior = FluxPriorReduxPipeline.from_pretrained(
        REDUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    ).to("cuda")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    )
    pipe.load_lora_weights(str(LORA_PATH), adapter_name="gh0st")
    pipe.set_adapters(["gh0st"], adapter_weights=[LORA_SCALE])
    pipe.enable_sequential_cpu_offload()
    print("  Pipelines ready.")

    done = 0
    for style, v in needed:
        suffix = version_suffix(v, versions)
        out = pure_dir / f"{style}{suffix}.png"
        if out.exists():
            continue

        print(f"  Generating pure {style}{suffix}...")
        anchor = load_image_rgb(get_representative_image(style))
        prior  = pipe_prior(image=anchor)

        generator = torch.Generator("cuda").manual_seed(SEEDS[v])
        result = pipe(
            prompt_embeds=prior.prompt_embeds,
            pooled_prompt_embeds=prior.pooled_prompt_embeds,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=GEN_SIZE,
            width=GEN_SIZE,
            generator=generator,
        ).images[0]
        result.save(out)
        done += 1
        print(f"    saved {out.name}")

    del pipe_prior, pipe
    torch.cuda.empty_cache()
    print(f"[Pure] Done — {done} pure frame(s) generated.")


# ---------------------------------------------------------------------------
# Phase: transition mashups
# ---------------------------------------------------------------------------
def generate_transitions(versions: int = 1):
    from diffusers import FluxPriorReduxPipeline, FluxPipeline

    # Count what needs doing
    total_needed = 0
    for set_a, set_b in TRANSITIONS:
        pair_dir = OUTPUT_DIR / f"{set_a}_x_{set_b}"
        for alpha in ALL_ALPHAS:
            pct_a = int(round(alpha * 100))
            pct_b = 100 - pct_a
            for v in range(versions):
                suffix = version_suffix(v, versions)
                fname = f"{pct_a:03d}{set_a}__{pct_b:03d}{set_b}{suffix}.png"
                if not (pair_dir / fname).exists():
                    total_needed += 1

    if total_needed == 0:
        print("[Transitions] All images already exist — skipping.")
        return

    print(f"\n[Transitions] Loading pipelines ({total_needed} image(s) to generate)...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipe_prior = FluxPriorReduxPipeline.from_pretrained(
        REDUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    ).to("cuda")
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL, torch_dtype=torch.bfloat16, cache_dir=str(MODELS_DIR)
    )
    pipe.load_lora_weights(str(LORA_PATH), adapter_name="gh0st")
    pipe.set_adapters(["gh0st"], adapter_weights=[LORA_SCALE])
    pipe.enable_sequential_cpu_offload()
    print("  Pipelines ready.")

    done = 0
    for set_a, set_b in TRANSITIONS:
        pair_dir = OUTPUT_DIR / f"{set_a}_x_{set_b}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Transition: {set_b} → {set_a}  ({len(ALL_ALPHAS) * versions} images)")

        try:
            anchor_a = load_image_rgb(get_representative_image(set_a))
            anchor_b = load_image_rgb(get_representative_image(set_b))
        except FileNotFoundError as e:
            print(f"    ERROR: {e} — skipping")
            continue

        print(f"    Encoding {set_a}...")
        prior_a = pipe_prior(image=anchor_a)
        print(f"    Encoding {set_b}...")
        prior_b = pipe_prior(image=anchor_b)

        for alpha in ALL_ALPHAS:
            pct_a = int(round(alpha * 100))
            pct_b = 100 - pct_a

            for v in range(versions):
                suffix = version_suffix(v, versions)
                fname    = f"{pct_a:03d}{set_a}__{pct_b:03d}{set_b}{suffix}.png"
                out_path = pair_dir / fname
                if out_path.exists():
                    print(f"    skip {fname}")
                    continue

                embeds = alpha * prior_a.prompt_embeds + (1 - alpha) * prior_b.prompt_embeds
                pooled = alpha * prior_a.pooled_prompt_embeds + (1 - alpha) * prior_b.pooled_prompt_embeds

                generator = torch.Generator("cuda").manual_seed(SEEDS[v])
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
                done += 1
                print(f"    saved {fname}")

    del pipe_prior, pipe
    torch.cuda.empty_cache()
    print(f"\n[Transitions] Done — {done} image(s) generated.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary():
    print("\n=== Summary ===")
    pure_dir = OUTPUT_DIR / "pure"
    pure_count = len(list(pure_dir.glob("*.png"))) if pure_dir.exists() else 0
    print(f"  Pure frames : {pure_count}")
    total_trans = 0
    for set_a, set_b in TRANSITIONS:
        pair_dir = OUTPUT_DIR / f"{set_a}_x_{set_b}"
        n = len(list(pair_dir.glob("*.png"))) if pair_dir.exists() else 0
        print(f"  {set_b:20s} → {set_a:20s}: {n}")
        total_trans += n
    print(f"  Total       : {pure_count + total_trans}")
    print(f"Output at: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLUX sequence generation")
    parser.add_argument("--versions", type=int, default=1, choices=[1, 2],
                        help="Number of versions per image (1 or 2, default: 1)")
    args = parser.parse_args()

    print("=== FLUX LoRA v2 — Sequence Generation ===")
    print(f"  LoRA  : {LORA_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Steps per transition: {len(ALL_ALPHAS) * args.versions} images "
          f"({len(ALL_ALPHAS)} ratios × {args.versions} version(s))")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_pure_frames(args.versions)
    generate_transitions(args.versions)
    print_summary()
    print("\nDONE")
