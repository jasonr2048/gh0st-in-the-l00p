# FLUX LoRA Training — Current State & Handoff to Claude Code

## What Cowork already built
All infrastructure is in place in this spike folder:

- `config.yaml` — ai-toolkit FLUX.1-dev LoRA training config, fully configured
- `run_training.sh` — uploads dataset via rsync, launches training in tmux on pod
- `monitor.sh` — tail training log, check GPU status, check samples
- `fetch_results.sh` — pulls .safetensors + sample images + log back locally
- `.runpod_env` — pod SSH host and HF token (gitignored)
- `.runpod_key` — SSH private key for RunPod pod (gitignored)

## What's NOT done yet
1. **Dataset not prepared** — `dataset/lora_training/` folder needs to be created
   with selected images + caption `.txt` files. See original BRIEF.md for details.
   Images are at:
   `/Users/jasonrobert/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/.shortcut-targets-by-id/1wVX93UBmrYHkO8xjnKxh429ya-a0rqk3/Gh0st in the Loop/dataset/prepared/`
   Captions should use `data/dataset_tags.md` as source.

2. **FLUX.1-dev terms** — jasonr2048 on HuggingFace needs to accept gated model terms at:
   https://huggingface.co/black-forest-labs/FLUX.1-dev

3. **Training not started** — once dataset is ready, run:
   ```bash
   bash spikes/flux_lora_training/run_training.sh
   ```
   This uploads dataset, launches training in tmux on pod, then exits.
   You can close your laptop. Training continues on RunPod (~40 min on RTX 4090).

## How to monitor (from terminal, any time)
```bash
bash spikes/flux_lora_training/monitor.sh           # tail live log
bash spikes/flux_lora_training/monitor.sh --status  # quick status + GPU
```

## How to fetch results when done
```bash
bash spikes/flux_lora_training/fetch_results.sh
```
Results land in `spikes/flux_lora_training/output/` — .safetensors + sample images.
**Then stop the RunPod pod to stop billing.**

## RunPod pod details
- Host: in `.runpod_env`
- GPU: RTX 4090, ~$0.44/hr
- Template: ai-toolkit (ostris) — FLUX.1-dev + ai-toolkit pre-installed
- Billing stops when pod is stopped/terminated on RunPod dashboard

## Why Claude Code instead of Cowork
Cowork's VM has no outbound internet (SSH blocked by egress proxy).
Claude Code has native SSH support in the desktop app and can connect directly to the pod.

## Task for Claude Code
1. Prepare `dataset/lora_training/` — select ~100 images from `dataset/prepared/`,
   generate caption .txt files using `data/dataset_tags.md` as source
2. Confirm HuggingFace terms accepted for jasonr2048
3. Run `bash spikes/flux_lora_training/run_training.sh`
4. Confirm tmux session launched successfully and training is running
5. Come back in ~40 min, run `bash spikes/flux_lora_training/fetch_results.sh`
6. Report back: sample images, whether A's face is recognisable, total cost
