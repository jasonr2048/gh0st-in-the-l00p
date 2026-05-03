# RunPod Network Volume setup

Persistent storage that survives pod stops. Pay ~$0.07/GB/month
(a 10 GB volume = ~$0.70/month). Pod itself costs nothing while stopped.

## One-time setup

### 1. Create the volume
- RunPod console → **Storage** → **New Network Volume**
- Name: `gh0st-workspace`
- Size: `10 GB` (enough for LoRA, stills, future experiments)
- Region: pick the same region your pods typically land in (EU-RO-1 or similar)
- Click **Create**

### 2. First pod: migrate existing data
Spin up a pod **with the volume attached**:
- Templates → any CUDA template (RunPod Pytorch is fine)
- Before deploying: expand **Storage** → attach `gh0st-workspace` at `/workspace`
- Deploy

SSH in and run:

```bash
# Verify volume is mounted
df -h /workspace

# Re-download the LoRA (already chunked — reuse deploy script or manual wget)
# Copy chunks from catbox URLs saved in your notes, or re-upload from local:
#   scp -i ~/.ssh/id_ed25519 -P <port> gh0st_flux_lora_v2.safetensors root@<host>:/workspace/

# Create the permanent directory layout
mkdir -p /workspace/lora
mkdir -p /workspace/output/gh0st_exhibition_v1
mkdir -p /workspace/source/normalized

# Move LoRA into place
mv /workspace/gh0st_flux_lora_v2.safetensors /workspace/lora/
```

Upload source images from local (run on your Mac):
```bash
rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p <port>" \
  spikes/flux_lora_training/exhibition_source/normalized/ \
  root@<host>:/workspace/source/normalized/
```

### 3. Update paths in generate_exhibition.py
The script currently expects `/workspace/` paths. With the volume mounted
at `/workspace` these are unchanged — nothing to edit.

Check the key constants at the top of `generate_exhibition.py`:
```python
LORA_PATH   = Path("/workspace/lora/gh0st_flux_lora_v2.safetensors")
SOURCE_DIR  = Path("/workspace/source/normalized")
OUTPUT_BASE = Path("/workspace/output")
```

If they don't match, update them once and commit.

### 4. Update setup_exhibition.sh
The deploy script currently downloads the LoRA and source images on every pod start.
With a volume, skip those steps — just launch generation directly:

```bash
# In tmux, just run:
cd /workspace
python generate_exhibition.py --output gh0st_exhibition_v1 --overwrite
```

Or update `setup_exhibition.sh` to detect if the LoRA already exists and skip download:
```bash
if [ ! -f "$LORA_PATH" ]; then
    echo "LoRA not found, downloading..."
    # ... download logic ...
fi
```

## Day-to-day workflow

1. **Start a pod** — attach `gh0st-workspace` at `/workspace` (takes ~10s to mount)
2. **Run generation** — stills, LoRA, sources all already there
3. **Stop the pod** — volume persists, you pay ~$0/hr
4. **Fetch results** — use `fetch_exhibition.sh` as before (stills are still in `/workspace/output/`)

## Attaching the volume to a new pod

Every time you create a pod:
- Expand **Storage** section in the pod config
- Select `gh0st-workspace` → mount at `/workspace`

That's it. Everything from the previous session is there.

## If you need to copy stills back locally

`fetch_exhibition.sh` already handles this via catbox. No change needed.
Or with rsync if you have a direct SSH port:
```bash
rsync -av -e "ssh -i ~/.ssh/id_ed25519 -p <port>" \
  root@<host>:/workspace/output/gh0st_exhibition_v1/ \
  spikes/flux_lora_training/output/gh0st_exhibition_v1/
```
