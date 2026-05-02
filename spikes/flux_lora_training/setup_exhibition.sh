#!/usr/bin/env bash
# Setup pod for exhibition generation + launch.
#
# Packs source images (per selection file) + generate_exhibition.py + LoRA,
# uploads to litterbox.catbox.moe, deploys to pod, launches in tmux.
#
# Usage:
#   bash spikes/flux_lora_training/setup_exhibition.sh
#   bash spikes/flux_lora_training/setup_exhibition.sh --selection selection_v2.txt --output gh0st_exhibition_v2
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

# ── Defaults (override via flags) ──────────────────────────────────────────
SELECTION_FILE="selection_v1.txt"
OUTPUT_NAME="gh0st_exhibition_v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --selection) SELECTION_FILE="$2"; shift 2 ;;
        --output)    OUTPUT_NAME="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DRIVE_FACES="$HOME/Library/CloudStorage/GoogleDrive-jorb2048@gmail.com/My Drive/Gh0st in the Loop/faces_exhibition"
SELECTION_PATH="$SPIKE_DIR/exhibition_source/$SELECTION_FILE"
LORA_FILE="$SPIKE_DIR/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors"
SCRIPT_FILE="$SPIKE_DIR/generate_exhibition.py"

echo "=== FLUX Exhibition Setup ==="
echo "  Selection : $SELECTION_FILE"
echo "  Output    : $OUTPUT_NAME"
echo "  Pod       : $POD_HOST"
echo ""

# ── Helpers ────────────────────────────────────────────────────────────────
catbox_upload() {
    local file="$1" label="$2" out="$3"
    echo "[upload] $label ($(du -sh "$file" | cut -f1))..."
    local url
    url=$(curl -s \
               --retry 3 --retry-delay 5 \
               -F "reqtype=fileupload" \
               -F "time=24h" \
               -F "fileToUpload=@${file}" \
               https://litterbox.catbox.moe/resources/internals/api.php)
    if [[ -z "$url" || "$url" != https* ]]; then
        echo "ERROR: catbox upload failed for $label — response: $url"
        exit 1
    fi
    echo "[upload] $label → $url"
    echo "$url" > "$out"
}

# ── [1] Pack source images listed in selection file ─────────────────────────
echo "=== [1/4] Packaging source images ==="
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

SOURCE_STAGING="$TMP_DIR/exhibition_source"
mkdir -p "$SOURCE_STAGING"

# Copy selection file
cp "$SELECTION_PATH" "$SOURCE_STAGING/$SELECTION_FILE"

# Copy all referenced source images, preserving category/filename structure
missing=0
while IFS= read -r line; do
    [[ -z "$line" || "$line" == "#"* ]] && continue
    src="$DRIVE_FACES/$line"
    cat_dir=$(dirname "$line")
    dst_dir="$SOURCE_STAGING/$cat_dir"
    mkdir -p "$dst_dir"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst_dir/"
    else
        echo "  WARNING: source not found: $src"
        missing=$((missing + 1))
    fi
done < "$SELECTION_PATH"

total_imgs=$(find "$SOURCE_STAGING" -type f ! -name "*.txt" | wc -l | tr -d ' ')
echo "  Packed $total_imgs source images ($missing missing)"
[[ $missing -gt 0 ]] && echo "  WARNING: $missing files not found — check Drive sync"

# Copy script
cp "$SCRIPT_FILE" "$TMP_DIR/generate_exhibition.py"

echo "  Tarballing source images + script..."
SOURCE_TAR="$TMP_DIR/exhibition_source.tar.gz"
tar -czf "$SOURCE_TAR" \
    --exclude="._*" \
    --exclude=".DS_Store" \
    -C "$TMP_DIR" exhibition_source generate_exhibition.py \
    2>/dev/null || true
echo "  Source tar: $(du -sh "$SOURCE_TAR" | cut -f1)"

echo "  Tarballing LoRA..."
LORA_TAR="$TMP_DIR/lora.tar.gz"
tar -czf "$LORA_TAR" -C "$(dirname "$LORA_FILE")" "$(basename "$LORA_FILE")"
echo "  LoRA tar: $(du -sh "$LORA_TAR" | cut -f1)"

# ── [2] Upload ──────────────────────────────────────────────────────────────
echo ""
echo "=== [2/4] Uploading to litterbox.catbox.moe ==="
URL_SOURCE_FILE="$TMP_DIR/url_source.txt"
URL_LORA_FILE="$TMP_DIR/url_lora.txt"

catbox_upload "$SOURCE_TAR" "exhibition source + script" "$URL_SOURCE_FILE"
catbox_upload "$LORA_TAR"   "LoRA weights"              "$URL_LORA_FILE"

URL_SOURCE=$(cat "$URL_SOURCE_FILE")
URL_LORA=$(cat "$URL_LORA_FILE")

echo ""
echo "  source URL : $URL_SOURCE"
echo "  LoRA URL   : $URL_LORA"

# ── [3] Deploy & launch on pod ─────────────────────────────────────────────
echo ""
echo "=== [3/4] Deploying to pod and launching ==="

EXPECT_SCRIPT=$(mktemp /tmp/setup_exh_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 600
set key         [lindex \$argv 0]
set host        [lindex \$argv 1]
set url_source  [lindex \$argv 2]
set url_lora    [lindex \$argv 3]
set hf_token    [lindex \$argv 4]
set sel_file    [lindex \$argv 5]
set out_name    [lindex \$argv 6]

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

# Directories
send "mkdir -p /workspace/exhibition_source /workspace/output; echo DIRS_OK\r"
ok "DIRS_OK"

# Download source tar
send "echo 'Downloading exhibition source...' && wget -q '\$url_source' -O /tmp/exhibition_source.tar.gz && echo DL_SOURCE_OK\r"
set timeout 300
ok "DL_SOURCE_OK"
set timeout 600

# Extract — script lands at /workspace/generate_exhibition.py, images at /workspace/exhibition_source/
send "tar -xzf /tmp/exhibition_source.tar.gz -C /workspace && echo EXTRACT_OK\r"
ok "EXTRACT_OK"

# Verify
send "ls /workspace/generate_exhibition.py && echo \$(find /workspace/exhibition_source -type f ! -name '*.txt' | wc -l) source images && echo VERIFY_SOURCE_OK\r"
ok "VERIFY_SOURCE_OK"

# LoRA — only download if not already present
send "if [ -f /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors ]; then echo LORA_EXISTS; else echo LORA_NEED_DL; fi\r"
expect {
    "LORA_EXISTS" {
        expect -re {#\s}
        send "echo LORA_OK\r"
        ok "LORA_OK"
    }
    "LORA_NEED_DL" {
        expect -re {#\s}
        send "mkdir -p /workspace/output/gh0st_flux_lora_v2 && wget -q '\$url_lora' -O /tmp/lora.tar.gz && tar -xzf /tmp/lora.tar.gz -C /workspace/output/gh0st_flux_lora_v2/ && echo LORA_OK\r"
        set timeout 600
        ok "LORA_OK"
    }
    timeout { puts stderr "TIMEOUT checking LoRA"; exit 1 }
}

send "ls -lh /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo LORA_VERIFY_OK\r"
ok "LORA_VERIFY_OK"

# HF token
send "mkdir -p /root/.huggingface /root/.cache/huggingface && printf '%s' '\$hf_token' > /root/.huggingface/token && printf '%s' '\$hf_token' > /root/.cache/huggingface/token && export HF_TOKEN='\$hf_token'; echo TOKEN_OK\r"
ok "TOKEN_OK"

# Clean macOS xattr dirs
send "find /workspace/exhibition_source -name '._*' -delete 2>/dev/null; echo CLEAN_OK\r"
ok "CLEAN_OK"

# Kill any old session, launch fresh
send "tmux kill-session -t exhibition 2>/dev/null || true; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\s}

send "tmux new-session -d -s exhibition 'cd /workspace && HF_TOKEN='\$hf_token' python generate_exhibition.py --selection \$sel_file --output \$out_name 2>&1 | tee /workspace/exhibition.log; echo GENERATION_COMPLETE >> /workspace/exhibition.log'; echo LAUNCH_OK\r"
ok "LAUNCH_OK"

send "sleep 5 && tail -15 /workspace/exhibition.log 2>/dev/null || echo '(log not started yet)'; echo TAIL_OK\r"
set timeout 30
expect { "TAIL_OK" {} timeout {} }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" \
    "$KEY" "$POD_HOST" "$URL_SOURCE" "$URL_LORA" \
    "$HF_TOKEN" "$SELECTION_FILE" "$OUTPUT_NAME" 2>&1 | \
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
echo "=== [4/4] Exhibition generation launched ==="
echo "Monitor : bash spikes/flux_lora_training/monitor_exhibition.sh"
echo "Fetch   : bash spikes/flux_lora_training/fetch_exhibition.sh --output $OUTPUT_NAME"
