#!/usr/bin/env bash
# Driver: upload dataset + launch ai-toolkit FLUX LoRA training on RunPod in tmux.
# Exits after launch; training continues on the pod (~30-45 min on RTX 4090).
# Usage:  bash spikes/flux_lora_training/run_training.sh
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"

KEY="$SPIKE_DIR/.runpod_key"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30)

# shellcheck disable=SC2016
: "${HF_TOKEN:?HF_TOKEN missing from .runpod_env}"
: "${POD_HOST:?POD_HOST missing from .runpod_env}"

DATASET="$HOME/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/.shortcut-targets-by-id/1wVX93UBmrYHkO8xjnKxh429ya-a0rqk3/Gh0st in the Loop/dataset/lora_training"

if [[ ! -d "$DATASET" ]]; then
  echo "[FATAL] Dataset folder not found: $DATASET" >&2
  exit 1
fi

img_count=$(find "$DATASET" -maxdepth 1 -iname '*.png' | wc -l | tr -d ' ')
txt_count=$(find "$DATASET" -maxdepth 1 -iname '*.txt' | wc -l | tr -d ' ')
echo "[info] dataset: $img_count images, $txt_count captions"

echo "[1/5] SSH smoke test..."
ssh "${SSH_OPTS[@]}" "$POD_HOST" 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && mkdir -p /workspace/lora_training /workspace/config /workspace/output'

echo "[2/5] Uploading dataset (rsync)..."
rsync -av --delete --progress -e "ssh ${SSH_OPTS[*]}" "$DATASET/" "$POD_HOST:/workspace/lora_training/"

echo "[3/5] Uploading config.yaml..."
scp "${SSH_OPTS[@]}" "$SPIKE_DIR/config.yaml" "$POD_HOST:/workspace/config/config.yaml"

echo "[4/5] Installing tmux, huggingface-cli login, launching training..."
ssh "${SSH_OPTS[@]}" "$POD_HOST" "HF_TOKEN='$HF_TOKEN' bash -s" <<'REMOTE'
set -euo pipefail
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

# Ensure ai-toolkit is present
if [[ ! -d /workspace/ai-toolkit ]]; then
  echo "[warn] /workspace/ai-toolkit missing — cloning fresh"
  cd /workspace && git clone https://github.com/ostris/ai-toolkit.git
  cd ai-toolkit && git submodule update --init --recursive
  pip install -q -r requirements.txt
fi

# HF login (needed for FLUX.1-dev gated repo)
python -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

# Write runner inside pod (HF_TOKEN baked via env)
cat > /workspace/run.sh <<RUNNER
#!/usr/bin/env bash
set -e
export HF_TOKEN='$HF_TOKEN'
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ai-toolkit
python run.py /workspace/config/config.yaml 2>&1 | tee /workspace/training.log
RUNNER
chmod +x /workspace/run.sh

# Kill any prior session so reruns are clean
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train '/workspace/run.sh; echo; echo "[DONE] exit=$?"; sleep 600'
sleep 2
tmux ls
echo '[ok] training launched in tmux session "train"'
REMOTE

echo "[5/5] done."
echo "  monitor: bash $SPIKE_DIR/monitor.sh"
echo "  fetch:   bash $SPIKE_DIR/fetch_results.sh   (after ~40 min)"
