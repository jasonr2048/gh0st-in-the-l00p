#!/usr/bin/env bash
# Full setup for fresh pod + launch sequence generation.
# Uploads dataset + generate_sequence.py + LoRA to litterbox.catbox.moe,
# then connects to pod, wgets everything, and launches in tmux.
#
# Usage: bash spikes/flux_lora_training/setup_sequence.sh [--versions 2]
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

VERSIONS="${1:-1}"
if [[ "${1:-}" == "--versions" ]]; then
    VERSIONS="${2:-1}"
fi

DRIVE_DATASET="$HOME/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/My Drive/Gh0st in the Loop/dataset/lora_training_v2"
LORA_FILE="$SPIKE_DIR/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors"
SCRIPT_FILE="$SPIKE_DIR/generate_sequence.py"

catbox_upload() {
  local file="$1" label="$2" out="$3"
  echo "[upload] $label ($(du -sh "$file" | cut -f1))..."
  local url
  url=$(curl -s \
             -F "reqtype=fileupload" \
             -F "time=24h" \
             -F "fileToUpload=@${file}" \
             https://litterbox.catbox.moe/resources/internals/api.php)
  if [[ -z "$url" || "$url" != https* ]]; then
    echo "ERROR: upload failed for $label — response: $url"
    exit 1
  fi
  echo "[upload] $label → $url"
  echo "$url" > "$out"
}

echo "=== [1/4] Packaging files ==="
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "  Tarballing dataset + generate_sequence.py..."
DATASET_TAR="$TMP_DIR/dataset_and_script.tar.gz"
cp "$SCRIPT_FILE" "$TMP_DIR/generate_sequence.py"
tar -czf "$DATASET_TAR" \
    --exclude="._*" \
    --exclude=".DS_Store" \
    -C "$TMP_DIR" generate_sequence.py \
    -C "$DRIVE_DATASET/.." lora_training_v2 \
    2>/dev/null || true
echo "  Dataset tar: $(du -sh "$DATASET_TAR" | cut -f1)"

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
echo "  dataset URL : $URL_DATASET"
echo "  LoRA URL    : $URL_LORA"

echo ""
echo "=== [3/4] Setting up pod and launching ==="

EXPECT_SCRIPT=$(mktemp /tmp/setup_seq_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 600
set key          [lindex \$argv 0]
set host         [lindex \$argv 1]
set url_dataset  [lindex \$argv 2]
set url_lora     [lindex \$argv 3]
set hf_token     [lindex \$argv 4]
set versions     [lindex \$argv 5]

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

send "mkdir -p /workspace/dataset /workspace/output/gh0st_flux_lora_v2; echo DIRS_OK\r"
ok "DIRS_OK"

send "echo 'Downloading dataset+script...' && wget -q '\$url_dataset' -O /tmp/dataset_and_script.tar.gz && echo DL_DATASET_OK\r"
set timeout 300
ok "DL_DATASET_OK"
set timeout 600

send "tar -xzf /tmp/dataset_and_script.tar.gz -C /workspace && mv /workspace/lora_training_v2 /workspace/dataset/ 2>/dev/null || true; echo EXTRACT_OK\r"
ok "EXTRACT_OK"

send "echo '--- dataset sets:' && ls /workspace/dataset/lora_training_v2/ | wc -l && ls /workspace/generate_sequence.py && echo VERIFY_OK\r"
ok "VERIFY_OK"

send "echo 'Downloading LoRA...' && wget -q '\$url_lora' -O /tmp/lora.tar.gz && echo DL_LORA_OK\r"
set timeout 600
ok "DL_LORA_OK"

send "tar -xzf /tmp/lora.tar.gz -C /workspace/output/gh0st_flux_lora_v2/ && echo LORA_OK\r"
ok "LORA_OK"

send "ls -lh /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo LORA_VERIFY_OK\r"
ok "LORA_VERIFY_OK"

# Write HF token to both old and new locations, and export as env var
send "mkdir -p /root/.huggingface /root/.cache/huggingface && printf '%s' '\$hf_token' > /root/.huggingface/token && printf '%s' '\$hf_token' > /root/.cache/huggingface/token && export HF_TOKEN='\$hf_token'; echo TOKEN_OK\r"
ok "TOKEN_OK"

# Remove any macOS xattr dirs
send "rm -rf /workspace/dataset/lora_training_v2/._* 2>/dev/null; echo CLEAN_OK\r"
ok "CLEAN_OK"

# Kill any old session, launch fresh
send "tmux kill-session -t sequence 2>/dev/null || true; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\s}

send "tmux new-session -d -s sequence 'cd /workspace && HF_TOKEN='\$hf_token' python generate_sequence.py --versions \$versions 2>&1 | tee /workspace/sequence.log; echo GENERATION_COMPLETE >> /workspace/sequence.log'; echo LAUNCH_OK\r"
ok "LAUNCH_OK"

send "sleep 5 && tail -10 /workspace/sequence.log 2>/dev/null || echo '(log not started yet)'; echo TAIL_OK\r"
set timeout 30
expect { "TAIL_OK" {} timeout {} }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$URL_DATASET" "$URL_LORA" "$HF_TOKEN" "$VERSIONS" 2>&1 | \
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
echo "=== [4/4] Sequence generation launched ==="
echo "Monitor with: bash spikes/flux_lora_training/monitor_sequence.sh"
echo "Fetch with  : bash spikes/flux_lora_training/fetch_sequence.sh"
