#!/usr/bin/env bash
# Setup pod for exhibition generation + launch.
#
# Volume-aware: LoRA and source images are only uploaded if not already
# present on /workspace (network volume). generate_exhibition.py is always
# refreshed. On first use this uploads everything; subsequent runs skip
# the large uploads and just deploy the latest script + launch.
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
SELECTION_FILE="selection_v1_normalized.txt"
OUTPUT_NAME="gh0st_exhibition_v1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --selection) SELECTION_FILE="$2"; shift 2 ;;
        --output)    OUTPUT_NAME="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NORMALIZED_DIR="$SPIKE_DIR/exhibition_source/normalized"
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

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# ── [1] Check what's already on the volume ─────────────────────────────────
echo "=== [1/4] Checking what's already on the volume ==="

PROBE_SCRIPT=$(mktemp /tmp/probe_XXXXXX.expect)
trap 'rm -f "$PROBE_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$PROBE_SCRIPT" << 'TCEOF'
#!/usr/bin/expect -f
set key  [lindex $argv 0]
set host [lindex $argv 1]
set timeout 30

spawn ssh -t \
    -i $key \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    $host

expect -re {#\s}
send "test -f /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo LORA_EXISTS || echo LORA_MISSING\r"
expect -re {#\s}
send "test -d /workspace/exhibition_source && find /workspace/exhibition_source -name '*.png' | wc -l | tr -d ' ' && echo SOURCES_CHECKED\r"
expect { "SOURCES_CHECKED" {} timeout {} }
expect -re {#\s}
send "exit\r"
expect eof
TCEOF
chmod +x "$PROBE_SCRIPT"

PROBE_OUT=$(expect "$PROBE_SCRIPT" "$KEY" "$POD_HOST" 2>&1 | \
    sed 's/\x1b\[[0-9;]*[mKHF]//g' | tr -d '\r')

LORA_ON_VOLUME=false
SOURCES_ON_VOLUME=false

if echo "$PROBE_OUT" | grep -q "LORA_EXISTS"; then
    LORA_ON_VOLUME=true
    echo "  LoRA        : already on volume ✓"
else
    echo "  LoRA        : not found — will upload"
fi

SOURCE_COUNT=$(echo "$PROBE_OUT" | grep -E '^\s*[0-9]+\s*$' | tr -d ' ' | head -1)
if [[ -n "$SOURCE_COUNT" && "$SOURCE_COUNT" -gt 0 ]]; then
    SOURCES_ON_VOLUME=true
    echo "  Sources     : $SOURCE_COUNT images already on volume ✓"
else
    echo "  Sources     : not found — will upload"
fi

# ── [2] Package and upload what's missing ──────────────────────────────────
echo ""
echo "=== [2/4] Packaging and uploading ==="

# Always upload the latest script + selection file (small, may have changed)
SCRIPT_TAR="$TMP_DIR/script.tar.gz"
cp "$SCRIPT_FILE" "$TMP_DIR/generate_exhibition.py"
cp "$SELECTION_PATH" "$TMP_DIR/$SELECTION_FILE"
tar -czf "$SCRIPT_TAR" -C "$TMP_DIR" "generate_exhibition.py" "$SELECTION_FILE"
URL_SCRIPT_FILE="$TMP_DIR/url_script.txt"
catbox_upload "$SCRIPT_TAR" "generate_exhibition.py + selection" "$URL_SCRIPT_FILE"
URL_SCRIPT=$(cat "$URL_SCRIPT_FILE")

# Source images — only if not on volume
URL_SOURCE=""
if [[ "$SOURCES_ON_VOLUME" == "false" ]]; then
    SOURCE_STAGING="$TMP_DIR/exhibition_source"
    mkdir -p "$SOURCE_STAGING"
    cp "$SELECTION_PATH" "$SOURCE_STAGING/$SELECTION_FILE"
    missing=0
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == "#"* ]] && continue
        src="$NORMALIZED_DIR/$line"
        cat_dir=$(dirname "$line")
        mkdir -p "$SOURCE_STAGING/$cat_dir"
        if [[ -f "$src" ]]; then
            cp "$src" "$SOURCE_STAGING/$cat_dir/"
        else
            echo "  WARNING: source not found: $src"
            missing=$((missing + 1))
        fi
    done < "$SELECTION_PATH"
    total_imgs=$(find "$SOURCE_STAGING" -type f ! -name "*.txt" | wc -l | tr -d ' ')
    echo "  Packed $total_imgs source images ($missing missing)"
    SOURCE_TAR="$TMP_DIR/exhibition_source.tar.gz"
    tar -czf "$SOURCE_TAR" --exclude="._*" --exclude=".DS_Store" \
        -C "$TMP_DIR" exhibition_source 2>/dev/null || true
    URL_SOURCE_FILE="$TMP_DIR/url_source.txt"
    catbox_upload "$SOURCE_TAR" "exhibition source images" "$URL_SOURCE_FILE"
    URL_SOURCE=$(cat "$URL_SOURCE_FILE")
fi

# LoRA — only if not on volume
URL_LORA=""
if [[ "$LORA_ON_VOLUME" == "false" ]]; then
    LORA_TAR="$TMP_DIR/lora.tar.gz"
    tar -czf "$LORA_TAR" -C "$(dirname "$LORA_FILE")" "$(basename "$LORA_FILE")"
    URL_LORA_FILE="$TMP_DIR/url_lora.txt"
    catbox_upload "$LORA_TAR" "LoRA weights" "$URL_LORA_FILE"
    URL_LORA=$(cat "$URL_LORA_FILE")
fi

# ── [3] Deploy & launch on pod ─────────────────────────────────────────────
echo ""
echo "=== [3/4] Deploying to pod and launching ==="

EXPECT_SCRIPT=$(mktemp /tmp/setup_exh_XXXXXX.expect)
trap 'rm -f "$PROBE_SCRIPT" "$EXPECT_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 600
set key          [lindex \$argv 0]
set host         [lindex \$argv 1]
set url_script   [lindex \$argv 2]
set url_source   [lindex \$argv 3]
set url_lora     [lindex \$argv 4]
set hf_token     [lindex \$argv 5]
set sel_file     [lindex \$argv 6]
set out_name     [lindex \$argv 7]

proc wait_for {marker} {
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

send "mkdir -p /workspace/exhibition_source /workspace/output; echo DIRS_OK\r"
wait_for "DIRS_OK"

# Always deploy latest script + selection file
send "wget -q '\$url_script' -O /tmp/script.tar.gz && tar -xzf /tmp/script.tar.gz -C /workspace && echo SCRIPT_OK\r"
set timeout 120
wait_for "SCRIPT_OK"
set timeout 600

# Source images — only if URL provided
if {"\$url_source" ne ""} {
    send "echo 'Downloading source images...' && wget -q '\$url_source' -O /tmp/exhibition_source.tar.gz && tar -xzf /tmp/exhibition_source.tar.gz -C /workspace && echo SOURCES_OK\r"
    set timeout 300
    wait_for "SOURCES_OK"
    set timeout 600
} else {
    send "echo 'Source images already on volume — skipping download'; echo SOURCES_OK\r"
    wait_for "SOURCES_OK"
}

# LoRA — only if URL provided
if {"\$url_lora" ne ""} {
    send "echo 'Downloading LoRA...' && mkdir -p /workspace/output/gh0st_flux_lora_v2 && wget -q '\$url_lora' -O /tmp/lora.tar.gz && tar -xzf /tmp/lora.tar.gz -C /workspace/output/gh0st_flux_lora_v2/ && echo LORA_OK\r"
    set timeout 600
    wait_for "LORA_OK"
} else {
    send "echo 'LoRA already on volume — skipping download'; echo LORA_OK\r"
    wait_for "LORA_OK"
}

send "ls -lh /workspace/generate_exhibition.py /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo VERIFY_OK\r"
wait_for "VERIFY_OK"

# HF token
send "mkdir -p /root/.huggingface /root/.cache/huggingface && printf '%s' '\$hf_token' > /root/.huggingface/token && printf '%s' '\$hf_token' > /root/.cache/huggingface/token && export HF_TOKEN='\$hf_token'; echo TOKEN_OK\r"
wait_for "TOKEN_OK"

send "find /workspace/exhibition_source -name '._*' -delete 2>/dev/null; echo CLEAN_OK\r"
wait_for "CLEAN_OK"

# Kill any old session, launch fresh
send "tmux kill-session -t exhibition 2>/dev/null || true; sleep 1; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\s}

send "tmux new-session -d -s exhibition 'cd /workspace && HF_TOKEN='\$hf_token' python generate_exhibition.py --selection \$sel_file --output \$out_name 2>&1 | tee /workspace/exhibition.log; echo GENERATION_COMPLETE >> /workspace/exhibition.log'; echo LAUNCH_OK\r"
wait_for "LAUNCH_OK"

send "sleep 5 && tail -15 /workspace/exhibition.log 2>/dev/null || echo '(log not started yet)'; echo TAIL_OK\r"
set timeout 30
expect { "TAIL_OK" {} timeout {} }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" \
    "$KEY" "$POD_HOST" "$URL_SCRIPT" "$URL_SOURCE" "$URL_LORA" \
    "$HF_TOKEN" "$SELECTION_FILE" "$OUTPUT_NAME" 2>&1 | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | tr -d '\r' | \
  grep -Ev '^\[.2004|^spawn |Warning.*known_hosts|RUNPOD.IO|Enjoy your Pod|^--$|^exit$|Connection to.*closed|^$'

echo ""
echo "=== [4/4] Exhibition generation launched ==="
echo "Monitor : bash spikes/flux_lora_training/monitor_exhibition.sh"
echo "Fetch   : bash spikes/flux_lora_training/fetch_exhibition.sh --output $OUTPUT_NAME"
