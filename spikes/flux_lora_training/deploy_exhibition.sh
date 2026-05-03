#!/usr/bin/env bash
# One-shot deploy: upload a setup script to catbox, download + run it on pod.
# The pod-side script downloads source+LoRA in background, then launches generation.
# Use monitor_exhibition.sh to track progress after this returns.
#
# Usage:
#   bash spikes/flux_lora_training/deploy_exhibition.sh \
#     --source  https://litter.catbox.moe/XXXXX.gz \
#     --lora    https://litter.catbox.moe/YYYYY.gz
#
# Optional:
#   --output    gh0st_exhibition_v1 (default)
#   --selection selection_v1_normalized.txt (default)
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$HOME/.ssh/id_ed25519"

URL_SOURCE=""
URL_LORA=""
OUTPUT_NAME="gh0st_exhibition_v1"
SELECTION_FILE="selection_v1_normalized.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)    URL_SOURCE="$2";    shift 2 ;;
        --lora)      URL_LORA="$2";      shift 2 ;;
        --output)    OUTPUT_NAME="$2";   shift 2 ;;
        --selection) SELECTION_FILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$URL_SOURCE" || -z "$URL_LORA" ]]; then
    echo "ERROR: --source and --lora URLs required"
    echo "Usage: bash $0 --source URL --lora URL"
    exit 1
fi

echo "=== FLUX Exhibition Deploy ==="
echo "  Source URL : $URL_SOURCE"
echo "  LoRA URL   : $URL_LORA"
echo "  Output     : $OUTPUT_NAME"
echo "  Selection  : $SELECTION_FILE"
echo "  Pod        : $POD_HOST"
echo ""

# ── [1] Write pod-side setup script locally ───────────────────────────────────
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

POD_SCRIPT="$TMP_DIR/pod_setup.sh"
cat > "$POD_SCRIPT" << PODSCRIPT
#!/usr/bin/env bash
set -euo pipefail
LOG=/workspace/setup.log
mkdir -p /workspace
exec > >(tee -a "\$LOG") 2>&1

echo "[\$(date)] === Pod setup starting ==="
mkdir -p /workspace/exhibition_source /workspace/output

# Download source
echo "[\$(date)] Downloading source..."
wget -c -q --show-progress --progress=bar:force:noscroll \\
    "${URL_SOURCE}" -O /tmp/exhibition_source.tar.gz
echo "[\$(date)] Source: \$(ls -lh /tmp/exhibition_source.tar.gz | awk '{print \$5}')"

echo "[\$(date)] Extracting source..."
tar -xzf /tmp/exhibition_source.tar.gz -C /workspace
find /workspace/exhibition_source -name '._*' -delete 2>/dev/null || true
count=\$(find /workspace/exhibition_source -type f | wc -l | tr -d ' ')
echo "[\$(date)] Extracted \${count} source files + script."

# Download LoRA if needed
LORA_PATH=/workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors
if [[ -f "\${LORA_PATH}" ]]; then
    echo "[\$(date)] LoRA already present: \$(ls -lh \${LORA_PATH} | awk '{print \$5}')"
else
    echo "[\$(date)] Downloading LoRA..."
    mkdir -p /workspace/output/gh0st_flux_lora_v2
    wget -c -q --show-progress --progress=bar:force:noscroll \\
        "${URL_LORA}" -O /tmp/lora.tar.gz
    echo "[\$(date)] LoRA archive: \$(ls -lh /tmp/lora.tar.gz | awk '{print \$5}')"
    echo "[\$(date)] Extracting LoRA..."
    tar -xzf /tmp/lora.tar.gz -C /workspace/output/gh0st_flux_lora_v2/
    echo "[\$(date)] LoRA: \$(ls -lh \${LORA_PATH} | awk '{print \$5}')"
fi

# HF token
mkdir -p /root/.huggingface /root/.cache/huggingface
printf '%s' '${HF_TOKEN}' > /root/.huggingface/token
printf '%s' '${HF_TOKEN}' > /root/.cache/huggingface/token
echo "[\$(date)] HF token written."

# Launch exhibition generation in its own tmux session
echo "[\$(date)] Launching exhibition generation..."
tmux kill-session -t exhibition 2>/dev/null || true
tmux new-session -d -s exhibition \\
    "cd /workspace && HF_TOKEN='${HF_TOKEN}' python generate_exhibition.py --selection ${SELECTION_FILE} --output ${OUTPUT_NAME} 2>&1 | tee /workspace/exhibition.log; echo GENERATION_COMPLETE >> /workspace/exhibition.log"

echo "[\$(date)] Generation launched in tmux:exhibition."
echo "[\$(date)] === Setup complete ==="
PODSCRIPT

chmod +x "$POD_SCRIPT"
echo "  Pod setup script: $(wc -l < "$POD_SCRIPT") lines"

# ── [2] Upload setup script to catbox ─────────────────────────────────────────
echo ""
echo "=== [1/2] Uploading setup script to catbox ==="
URL_SETUP=$(curl -s \
    --retry 3 --retry-delay 5 \
    -F "reqtype=fileupload" \
    -F "time=24h" \
    -F "fileToUpload=@${POD_SCRIPT}" \
    https://litterbox.catbox.moe/resources/internals/api.php)

if [[ -z "$URL_SETUP" || "$URL_SETUP" != https* ]]; then
    echo "ERROR: catbox upload failed — $URL_SETUP"
    exit 1
fi
echo "  Setup script → $URL_SETUP"

# ── [3] SSH to pod: download + run in tmux ────────────────────────────────────
echo ""
echo "=== [2/2] Deploying to pod ==="

EXPECT_SCRIPT=$(mktemp /tmp/deploy_exh_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"; rm -rf "$TMP_DIR"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 60
set key        [lindex \$argv 0]
set host       [lindex \$argv 1]
set url_setup  [lindex \$argv 2]

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

# Kill any previous sessions
send "tmux kill-session -t setup 2>/dev/null; tmux kill-session -t exhibition 2>/dev/null; true; echo KILL_OK\r"
expect { "KILL_OK" {} timeout {} }
expect -re {#\s}

# Download setup script (fast — tiny file)
send "mkdir -p /workspace && wget -q '\$url_setup' -O /tmp/pod_setup.sh && chmod +x /tmp/pod_setup.sh && echo DL_SETUP_OK\r"
ok "DL_SETUP_OK"

# Launch in tmux
send "tmux new-session -d -s setup 'bash /tmp/pod_setup.sh'; echo LAUNCH_OK\r"
ok "LAUNCH_OK"

# Quick check
send "sleep 3 && tmux list-sessions 2>/dev/null && echo SESSIONS_OK\r"
set timeout 10
expect { "SESSIONS_OK" {} timeout {} }
expect -re {#\s}

send "cat /workspace/setup.log 2>/dev/null || echo '(log not started yet)'; echo PEEK_OK\r"
set timeout 15
expect { "PEEK_OK" {} timeout {} }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$URL_SETUP" 2>&1 | \
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
echo "=== Setup script running in pod tmux session 'setup' ==="
echo "  Downloads + generation launch happen autonomously on the pod."
echo "  Check with: bash spikes/flux_lora_training/monitor_exhibition.sh"
echo "  (tmux:setup will show download progress; tmux:exhibition shows generation)"
