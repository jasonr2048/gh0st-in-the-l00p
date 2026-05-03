#!/usr/bin/env bash
# Setup pod for exhibition generation + launch.
#
# Volume-aware: probes the pod to check what's already on /workspace
# before uploading. LoRA (328MB) is split into 50MB chunks for catbox.
# On subsequent runs with the same volume, only the script is re-uploaded.
#
# Usage:
#   bash spikes/flux_lora_training/setup_exhibition.sh
#   bash spikes/flux_lora_training/setup_exhibition.sh --selection selection_v2.txt --output gh0st_exhibition_v2
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

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
LORA_POD_PATH="/workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors"
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
    url=$(curl -s --retry 3 --retry-delay 5 \
               -F "reqtype=fileupload" -F "time=24h" \
               -F "fileToUpload=@${file}" \
               https://litterbox.catbox.moe/resources/internals/api.php)
    if [[ -z "$url" || "$url" != https* ]]; then
        echo "ERROR: catbox upload failed for $label — response: $url"; exit 1
    fi
    echo "[upload] $label → $url"
    echo "$url" > "$out"
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# ── [1] Probe pod: check what's on the volume ──────────────────────────────
# Note: PTY echoes commands back, so we use `echo MARKER_$?` — the $? is
# expanded on the pod AFTER execution, so echoed command text contains
# literal "$?" while actual output contains the exit code number.
echo "=== [1/4] Checking what's already on the volume ==="

PROBE_SCRIPT="$TMP_DIR/probe.expect"
cat > "$PROBE_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 30
spawn ssh -t -i [lindex \$argv 0] \\
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
    [lindex \$argv 1]
expect -re {#\\s}
send "test -f $LORA_POD_PATH; echo RESULT_LORA_\\\$?\r"
expect -re {#\\s}
send "COUNT=\\\$(find /workspace/exhibition_source -name '*.png' 2>/dev/null | wc -l | tr -d ' '); echo RESULT_SRC_\\\$COUNT\r"
expect -re {#\\s}
send "exit\r"
expect eof
TCEOF
chmod +x "$PROBE_SCRIPT"

PROBE_OUT=$(expect "$PROBE_SCRIPT" "$KEY" "$POD_HOST" 2>&1 \
    | sed 's/\x1b\[[0-9;]*[mKHF]//g' | tr -d '\r')

# RESULT_LORA_0 = file exists; RESULT_LORA_1 = missing
# Echoed command contains literal "$?" so won't match "RESULT_LORA_0"
LORA_ON_VOLUME=false
if echo "$PROBE_OUT" | grep -qF "RESULT_LORA_0"; then
    LORA_ON_VOLUME=true
    echo "  LoRA    : already on volume ✓"
else
    echo "  LoRA    : not found — will upload"
fi

# RESULT_SRC_52 = 52 sources found; RESULT_SRC_0 = none
SOURCE_COUNT=$(echo "$PROBE_OUT" | grep -oE 'RESULT_SRC_[0-9]+' | grep -oE '[0-9]+$' | tail -1 || true)
SOURCE_COUNT="${SOURCE_COUNT:-0}"
SOURCES_ON_VOLUME=false
if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    SOURCES_ON_VOLUME=true
    echo "  Sources : $SOURCE_COUNT images already on volume ✓"
else
    echo "  Sources : not found — will upload"
fi

# ── [2] Package and upload ─────────────────────────────────────────────────
echo ""
echo "=== [2/4] Packaging and uploading ==="

# Script + selection: always refresh (tiny, may have changed)
cp "$SCRIPT_FILE" "$TMP_DIR/generate_exhibition.py"
cp "$SELECTION_PATH" "$TMP_DIR/$SELECTION_FILE"
SCRIPT_TAR="$TMP_DIR/script.tar.gz"
tar -czf "$SCRIPT_TAR" -C "$TMP_DIR" "generate_exhibition.py" "$SELECTION_FILE"
URL_SCRIPT_F="$TMP_DIR/url_script.txt"
catbox_upload "$SCRIPT_TAR" "script + selection" "$URL_SCRIPT_F"
URL_SCRIPT=$(cat "$URL_SCRIPT_F")

# Sources: only if not on volume
URL_SOURCE=""
if [[ "$SOURCES_ON_VOLUME" == "false" ]]; then
    SOURCE_STAGING="$TMP_DIR/exhibition_source"
    mkdir -p "$SOURCE_STAGING"
    cp "$SELECTION_PATH" "$SOURCE_STAGING/$SELECTION_FILE"
    missing=0
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == "#"* ]] && continue
        src="$NORMALIZED_DIR/$line"
        mkdir -p "$SOURCE_STAGING/$(dirname "$line")"
        if [[ -f "$src" ]]; then
            cp "$src" "$SOURCE_STAGING/$(dirname "$line")/"
        else
            echo "  WARNING: source not found: $src"
            missing=$((missing + 1))
        fi
    done < "$SELECTION_PATH"
    echo "  Packed $(find "$SOURCE_STAGING" -type f ! -name "*.txt" | wc -l | tr -d ' ') sources ($missing missing)"
    SOURCE_TAR="$TMP_DIR/exhibition_source.tar.gz"
    tar -czf "$SOURCE_TAR" --exclude="._*" --exclude=".DS_Store" \
        -C "$TMP_DIR" exhibition_source 2>/dev/null || true
    URL_SOURCE_F="$TMP_DIR/url_source.txt"
    catbox_upload "$SOURCE_TAR" "source images" "$URL_SOURCE_F"
    URL_SOURCE=$(cat "$URL_SOURCE_F")
fi

# LoRA: chunked upload if not on volume
LORA_CHUNK_URLS=()
if [[ "$LORA_ON_VOLUME" == "false" && -f "$LORA_FILE" ]]; then
    CHUNK_DIR="$TMP_DIR/lora_chunks"
    mkdir -p "$CHUNK_DIR"
    split -b 50m "$LORA_FILE" "$CHUNK_DIR/chunk_"
    CHUNKS=($(ls "$CHUNK_DIR"/chunk_* | sort))
    echo "  Uploading LoRA in ${#CHUNKS[@]} chunks of 50MB..."
    for i in "${!CHUNKS[@]}"; do
        URL_F="$CHUNK_DIR/url_${i}.txt"
        catbox_upload "${CHUNKS[$i]}" "LoRA chunk $((i+1))/${#CHUNKS[@]}" "$URL_F"
        LORA_CHUNK_URLS+=("$(cat "$URL_F")")
        [[ $i -lt $((${#CHUNKS[@]}-1)) ]] && sleep 3
    done
elif [[ "$LORA_ON_VOLUME" == "false" ]]; then
    echo "  WARNING: LoRA not found locally at $LORA_FILE"
fi

# ── [3] Write pod-side setup script with URLs embedded ────────────────────
echo ""
echo "=== [3/4] Deploying to pod ==="

POD_SCRIPT="$TMP_DIR/pod_setup.sh"
cat > "$POD_SCRIPT" << PODEOF
#!/usr/bin/env bash
set -euo pipefail
echo "=== Pod setup starting ==="
mkdir -p /workspace/exhibition_source /workspace/output/gh0st_flux_lora_v2

# Script + selection (always)
wget -q "$URL_SCRIPT" -O /tmp/script.tar.gz
tar -xzf /tmp/script.tar.gz -C /workspace
echo "Script deployed."

# Sources
if [[ -n "$URL_SOURCE" ]]; then
    echo "Downloading source images..."
    wget -q "$URL_SOURCE" -O /tmp/src.tar.gz
    tar -xzf /tmp/src.tar.gz -C /workspace
    echo "Sources ready: \$(find /workspace/exhibition_source -name '*.png' | wc -l) images"
else
    echo "Sources already on volume (\$(find /workspace/exhibition_source -name '*.png' | wc -l) images)"
fi

# LoRA chunks
PODEOF

if [[ ${#LORA_CHUNK_URLS[@]} -gt 0 ]]; then
    cat >> "$POD_SCRIPT" << PODEOF2
echo "Downloading LoRA chunks in parallel..."
mkdir -p /tmp/lora_chunks
PODEOF2
    for i in "${!LORA_CHUNK_URLS[@]}"; do
        echo "wget -q '${LORA_CHUNK_URLS[$i]}' -O /tmp/lora_chunks/chunk_${i} &" >> "$POD_SCRIPT"
    done
    cat >> "$POD_SCRIPT" << 'PODEOF3'
wait
echo "Assembling LoRA..."
cat $(ls /tmp/lora_chunks/chunk_* | sort) > /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors
rm -rf /tmp/lora_chunks
echo "LoRA ready: $(du -sh /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors | cut -f1)"
PODEOF3
else
    cat >> "$POD_SCRIPT" << 'PODEOF4'
echo "LoRA already on volume."
PODEOF4
fi

cat >> "$POD_SCRIPT" << PODEOF5
# HF token
mkdir -p /root/.huggingface /root/.cache/huggingface
printf '%s' '$HF_TOKEN' > /root/.huggingface/token
printf '%s' '$HF_TOKEN' > /root/.cache/huggingface/token
echo "HF token set."

# Clean macOS xattr garbage
find /workspace/exhibition_source -name '._*' -delete 2>/dev/null || true

echo "=== Setup complete. Starting generation ==="
cd /workspace
HF_TOKEN='$HF_TOKEN' python generate_exhibition.py \\
    --selection "$SELECTION_FILE" \\
    --output "$OUTPUT_NAME" \\
    2>&1 | tee /workspace/exhibition.log
echo "GENERATION_COMPLETE" >> /workspace/exhibition.log
PODEOF5

# Upload the pod script
POD_SCRIPT_TAR="$TMP_DIR/pod_script.tar.gz"
tar -czf "$POD_SCRIPT_TAR" -C "$TMP_DIR" "pod_setup.sh"
URL_POD_SCRIPT_F="$TMP_DIR/url_pod_script.txt"
catbox_upload "$POD_SCRIPT_TAR" "pod setup script" "$URL_POD_SCRIPT_F"
URL_POD_SCRIPT=$(cat "$URL_POD_SCRIPT_F")

# ── [4] SSH in, launch setup in tmux ──────────────────────────────────────
DEPLOY_SCRIPT="$TMP_DIR/deploy.expect"
cat > "$DEPLOY_SCRIPT" << TCEOF2
#!/usr/bin/expect -f
set timeout 60
spawn ssh -t -i [lindex \$argv 0] \\
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \\
    -o ServerAliveInterval=30 [lindex \$argv 1]
expect -re {#\\s}

send "wget -q '$URL_POD_SCRIPT' -O /tmp/pod_script.tar.gz && tar -xzf /tmp/pod_script.tar.gz -C /tmp; echo DL_OK\r"
expect { "DL_OK" {} timeout { puts "timeout DL"; exit 1 } }
expect -re {#\\s}

send "chmod +x /tmp/pod_setup.sh; echo CHMOD_OK\r"
expect { "CHMOD_OK" {} timeout { puts "timeout CHMOD"; exit 1 } }
expect -re {#\\s}

send "tmux kill-session -t exhibition 2>/dev/null || true; sleep 1; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\\s}

send "tmux new-session -d -s exhibition '/tmp/pod_setup.sh'; echo LAUNCH_OK\r"
expect { "LAUNCH_OK" {} timeout { puts "timeout LAUNCH"; exit 1 } }
expect -re {#\\s}

send "sleep 5 && tail -8 /workspace/exhibition.log 2>/dev/null || echo '(not started yet)'; echo TAIL_OK\r"
set timeout 30
expect { "TAIL_OK" {} timeout {} }
expect -re {#\\s}

send "exit\r"
expect eof
TCEOF2
chmod +x "$DEPLOY_SCRIPT"

expect "$DEPLOY_SCRIPT" "$KEY" "$POD_HOST" 2>&1 \
    | sed 's/\x1b\[[0-9;]*[mKHF]//g' | tr -d '\r' \
    | grep -Ev '^\[.2004|^spawn |Warning.*known_hosts|RUNPOD.IO|Enjoy your Pod|^exit$|^$'

echo ""
echo "=== [4/4] Generation launched in tmux:exhibition ==="
echo "Monitor : bash spikes/flux_lora_training/monitor_exhibition.sh"
echo "Fetch   : bash spikes/flux_lora_training/fetch_exhibition.sh --output $OUTPUT_NAME"
