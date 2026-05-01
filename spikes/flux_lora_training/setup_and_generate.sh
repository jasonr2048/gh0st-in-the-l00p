#!/usr/bin/env bash
# Full setup for fresh pod + launch generation.
# Uploads dataset + generate.py + LoRA to litterbox.catbox.moe, then
# connects to pod and has it wget everything before launching in tmux.
# Usage: bash spikes/flux_lora_training/setup_and_generate.sh
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

DRIVE_DATASET="$HOME/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/My Drive/Gh0st in the Loop/dataset/lora_training_v2"
LORA_FILE="$SPIKE_DIR/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors"
GENERATE_PY="$SPIKE_DIR/generate.py"

catbox_upload() {
  # Usage: catbox_upload <file> <label> <out_var_file>
  # Writes URL to out_var_file; all status to stdout
  local file="$1" label="$2" out="$3"
  echo "[upload] $label ($(du -sh "$file" | cut -f1))..."
  local url
  url=$(curl -s \
             -F "reqtype=fileupload" \
             -F "time=24h" \
             -F "fileToUpload=@${file}" \
             https://litterbox.catbox.moe/resources/internals/api.php)
  if [[ -z "$url" || "$url" != https* ]]; then
    echo "ERROR: upload failed for $label"
    echo "  curl response: $url"
    exit 1
  fi
  echo "[upload] $label → $url"
  echo "$url" > "$out"
}

# ------------------------------------------------------------------
# Step 1: Build tars and upload
# ------------------------------------------------------------------
echo "=== [1/4] Packaging files for upload ==="
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Dataset + generate.py together (tiny)
echo "  Tarballing dataset + generate.py..."
DATASET_TAR="$TMP_DIR/dataset_and_script.tar.gz"
cp "$GENERATE_PY" "$TMP_DIR/generate.py"
# Use -C to avoid Drive path issues; exclude xattr junk
tar -czf "$DATASET_TAR" \
    --exclude="._*" \
    --exclude=".DS_Store" \
    -C "$TMP_DIR" generate.py \
    -C "$DRIVE_DATASET/.." lora_training_v2 \
    2>/dev/null || true
echo "  Dataset tar: $(du -sh "$DATASET_TAR" | cut -f1)"

# LoRA — tar separately (328 MB)
echo "  Tarballing LoRA weights..."
LORA_TAR="$TMP_DIR/lora.tar.gz"
tar -czf "$LORA_TAR" -C "$(dirname "$LORA_FILE")" "$(basename "$LORA_FILE")"
echo "  LoRA tar: $(du -sh "$LORA_TAR" | cut -f1)"

URL_DATASET_FILE="$TMP_DIR/url_dataset.txt"
URL_LORA_FILE="$TMP_DIR/url_lora.txt"

echo ""
echo "=== [2/4] Uploading to litterbox.catbox.moe ==="
catbox_upload "$DATASET_TAR" "dataset+script" "$URL_DATASET_FILE"
catbox_upload "$LORA_TAR"    "LoRA weights"   "$URL_LORA_FILE"

URL_DATASET=$(cat "$URL_DATASET_FILE")
URL_LORA=$(cat "$URL_LORA_FILE")

echo ""
echo "  dataset URL: $URL_DATASET"
echo "  LoRA URL:    $URL_LORA"

# ------------------------------------------------------------------
# Step 2: Connect to pod and set everything up
# ------------------------------------------------------------------
echo ""
echo "=== [3/4] Setting up pod and launching generation ==="

EXPECT_SCRIPT=$(mktemp /tmp/setup_gen_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 600
set key          [lindex \$argv 0]
set host         [lindex \$argv 1]
set url_dataset  [lindex \$argv 2]
set url_lora     [lindex \$argv 3]
set hf_token     [lindex \$argv 4]

proc ok {marker} {
    expect {
        \$marker {}
        timeout { puts stderr "TIMEOUT waiting for \$marker"; exit 1 }
    }
    expect -re {#\s}
}

spawn ssh -t \\
    -i \$key \\
    -o StrictHostKeyChecking=no \\
    -o UserKnownHostsFile=/dev/null \\
    -o ServerAliveInterval=30 \\
    \$host

expect -re {#\s}

# Create workspace dirs
send "mkdir -p /workspace/dataset /workspace/output/gh0st_flux_lora_v2; echo DIRS_OK\r"
ok "DIRS_OK"

# Download dataset + generate.py
send "echo 'Downloading dataset...' && wget -q '\$url_dataset' -O /tmp/dataset_and_script.tar.gz && echo DL_DATASET_OK\r"
set timeout 300
ok "DL_DATASET_OK"
set timeout 600

# Extract dataset
send "tar -xzf /tmp/dataset_and_script.tar.gz -C /workspace && mv /workspace/lora_training_v2 /workspace/dataset/ 2>/dev/null || true; echo EXTRACT_DATASET_OK\r"
ok "EXTRACT_DATASET_OK"

# Verify dataset and script
send "ls /workspace/dataset/lora_training_v2/ | wc -l && ls /workspace/generate.py && echo VERIFY_DATASET_OK\r"
ok "VERIFY_DATASET_OK"

# Download LoRA
send "echo 'Downloading LoRA...' && wget -q '\$url_lora' -O /tmp/lora.tar.gz && echo DL_LORA_OK\r"
set timeout 600
ok "DL_LORA_OK"
set timeout 600

# Extract LoRA
send "tar -xzf /tmp/lora.tar.gz -C /workspace/output/gh0st_flux_lora_v2/ && echo EXTRACT_LORA_OK\r"
ok "EXTRACT_LORA_OK"

# Verify LoRA
send "ls -lh /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo VERIFY_LORA_OK\r"
ok "VERIFY_LORA_OK"

# HuggingFace login (required for gated FLUX.1-dev model)
send "huggingface-cli login --token \$hf_token --add-to-git-credential 2>&1 | tail -1; echo HF_LOGIN_OK\r"
ok "HF_LOGIN_OK"

# Kill any old session, launch fresh
send "tmux kill-session -t generate 2>/dev/null || true; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\s}

send "tmux new-session -d -s generate 'cd /workspace && python generate.py 2>&1 | tee /workspace/generate.log; echo GENERATION_COMPLETE >> /workspace/generate.log'; echo LAUNCH_OK\r"
ok "LAUNCH_OK"

send "sleep 4 && tail -8 /workspace/generate.log 2>/dev/null || echo '(log not started yet)'; echo TAIL_OK\r"
set timeout 30
expect { "TAIL_OK" {} timeout {} }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$URL_DATASET" "$URL_LORA" "$HF_TOKEN" 2>&1 | \
  grep -v '^\[?2004' | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | \
  sed 's/\r//' | \
  grep -v '^$' | \
  grep -v '^spawn ' | \
  grep -v 'Warning:.*known_hosts' | \
  grep -v 'RUNPOD.IO' | \
  grep -v 'Enjoy your Pod' | \
  grep -v '^--$' | \
  grep -v 'exit$' | \
  grep -v 'Connection to.*closed'

echo ""
echo "=== [4/4] Generation launched ==="
echo "Monitor with: bash spikes/flux_lora_training/monitor_generate.sh"
