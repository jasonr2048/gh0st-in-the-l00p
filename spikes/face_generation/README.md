# Face Generation Spike

## Goal
Validate StyleGAN as the approach for generating synthetic, uncanny face-like visuals
in the style of A's makeup dataset. Explore latent space interpolation as the basis
for morphing between styles.

## Context
- Output is pre-rendered offline, not real-time — heavy models are acceptable
- Faces should be synthetic/abstract, not realistic — uncanny is the target aesthetic
- A's dataset: photos of her own face with various makeup styles (camouflage, tribal,
  hacker-inspired, TikTok filters, anime aesthetics)
- Final output: portrait-oriented video sequences for the display pipeline

## Approach

### Step 1 — Baseline generation (pretrained FFHQ model)
- Run StyleGAN2 or StyleGAN3 in Colab using a pretrained FFHQ checkpoint
- Generate a batch of faces, assess aesthetic quality and uncanniness
- No fine-tuning yet — just validate the tool and workflow

### Step 2 — Latent space interpolation (morphing)
- Interpolate between two random latent vectors
- Render as a sequence of frames
- Assess smoothness and visual quality of the morph
- This is the core mechanism for divergence between the two entities

### Step 3 — Fine-tuning on A's dataset (if Step 1/2 promising)
- Fine-tune pretrained model on A's makeup photos
- Assess whether the style transfers meaningfully
- Compare fine-tuned vs pretrained output aesthetically

## Tools
- **StyleGAN2-ADA** or **StyleGAN3** — pretrained FFHQ checkpoints available
- **Google Colab** — free tier sufficient for initial spike, Pro if needed
- **Replicate API** — fallback if Colab insufficient for fine-tuning pipeline

## Cost Controls
- Use Colab free tier first
- If moving to GCP, set budget alerts before starting any job
- Replicate charges per inference call — predictable and capped

## Success Criteria
- Can generate convincingly synthetic/uncanny face-like output
- Latent interpolation produces smooth, visually interesting morphs
- Pipeline is simple enough to run repeatedly for batch generation
- Output frames exportable as image sequences → video

## Output Format
- Portrait orientation (to match display pipeline)
- Image sequences → rendered to video offline
- Resolution TBC based on monitor specs (target 1080x1920 or similar portrait)

## Next Step
→ Feedback loop logic spike
