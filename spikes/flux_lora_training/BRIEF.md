# FLUX LoRA Training — Brief for Cowork

## Goal
Fine-tune a FLUX.1 LoRA on A's prepared face dataset to generate new images of A's face
in her various makeup styles. This runs in parallel to the optical flow interpolation
work — it's an exploration, not the primary deliverable.

## Context
See CLAUDE.md for full project context. Key points:
- A is the artistic collaborator. Her face with various extreme makeup styles is the subject.
- We have ~300 prepared images across 39 style sets in `dataset/prepared/` on Google Drive
- Images are 512x512 square crops, head-centred, black or plain backgrounds on many sets
- Output must be recognisably A's face — not generic synthetic faces

## Approach
Use **FLUX.1-dev + LoRA via ai-toolkit** on **RunPod**. This is the current
best-practice stack for face LoRA training with small datasets.

## Step 1 — Prepare the dataset
Select a subset of prepared images for training. Recommended:
- Use the cleanest sets (black background, face clearly visible)
- Good candidates: rhinestones, clown, blue_face, latex_skin, horror_drip,
  ceramic_mask, silicone_skull, bead_cage, gold_hardware, inflated_lips,
  black_crosses, crybabyglitch, red_white, facelift, mutate
- Aim for 80-150 images total — enough variety without overwhelming the model
- Copy selected images to a new folder: `dataset/lora_training/`

Images are at:
`/Users/jasonrobert/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/.shortcut-targets-by-id/1wVX93UBmrYHkO8xjnKxh429ya-a0rqk3/Gh0st in the Loop/dataset/prepared/`

## Step 2 — Caption the images
Each image needs a `.txt` caption file with the same filename.

Caption format:
```
sks person, [style description], portrait, black background
```

Where `sks` is the trigger word the model will learn to associate with A's face.

Style descriptions should describe the makeup/look. Examples:
- `sks person, rhinestone face makeup, dramatic portrait, black background`
- `sks person, blue face paint, avant-garde makeup, portrait, black background`
- `sks person, clown makeup, theatrical, portrait, black background`
- `sks person, ceramic mask aesthetic, sculptural makeup, portrait, black background`

Use the existing `data/dataset_tags.md` file as the source for style descriptions —
it already has tags and categories for all sets. Map each set's tags to caption text,
prepend `sks person,`, and generate the `.txt` files automatically.
Review a sample to check quality.

Save caption `.txt` files alongside the images in `dataset/lora_training/`.

## Step 3 — Provision RunPod instance
Go to https://www.runpod.io

1. Create account, add ~$10 credit
2. Go to Pods → Deploy
3. Search for template: **"AI Toolkit"** (by ostris) — official template
4. Select GPU: **RTX 4090** (community cloud, ~$0.44/hr) — good balance of speed/cost
5. Set volume size: 50GB minimum
6. Deploy and wait for it to start
7. Click Connect → Open JupyterLab

## Step 4 — Upload dataset
In JupyterLab file browser, upload the `dataset/lora_training/` folder
(images + caption .txt files) to `/workspace/`.

## Step 5 — Run training via ai-toolkit UI
In JupyterLab terminal:
```bash
cd /workspace/ai-toolkit
python flux_train_ui.py
```

This opens a browser UI. Configure:
- **Trigger word:** `sks`
- **Model:** FLUX.1-dev (requires HuggingFace token — accept terms at
  https://huggingface.co/black-forest-labs/FLUX.1-dev first)
- **Steps:** 2000-3000 (start with 2000)
- **LoRA rank:** 16 (start here, increase to 32 if results look generic)
- **Learning rate:** 0.0001
- **Sample prompts:** `sks person, rhinestone makeup, portrait` and
  `sks person, blue face paint, portrait` — to visualise progress during training

Start training. Sample images generate automatically every 250 steps.

## Step 6 — Monitor and download
Training takes ~30-45 minutes on RTX 4090 for 2000 steps.

Review sample images generated during training — they appear in the UI.
When training completes, download the `.safetensors` LoRA file from `/workspace/`.

**Important: shut down the pod when done to stop charges.**

## Step 7 — Test inference
Test the LoRA using a simple diffusers script. Try prompts like:
- `sks person, rhinestone makeup, portrait, black background` (seen style)
- `sks person, cyberpunk makeup, neon, portrait` (novel style)
- `sks person, tribal face paint, portrait` (novel style)

Goal: does it generate A's face convincingly in both seen and unseen styles?

## What to report back
- Sample images at training steps 500, 1000, 2000, 3000
- Whether A's face is recognisable vs generic
- Which styles transfer well vs poorly
- Total cost of the run
- Path to the .safetensors file for download

## Cost estimate
- Training 2000 steps on RTX 4090: ~30 min = ~$0.22
- Storage: ~$0.07/hr for 50GB volume
- Total: well under $2 for a full training run

## Notes
- HuggingFace account: jasonr2048 — accept FLUX.1-dev terms before starting:
  https://huggingface.co/black-forest-labs/FLUX.1-dev
- If ai-toolkit template not found, search "ostris" on RunPod templates page
- Keep pod running only while actively training — terminate when done
- This is an exploration — results may need multiple runs with different settings
