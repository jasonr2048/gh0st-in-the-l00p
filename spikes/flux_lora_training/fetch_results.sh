#!/usr/bin/env bash
# Pull LoRA + sample images + log from RunPod pod to local output/.
# RunPod gateway requires PTY (rsync/scp blocked) — uses tar-over-catbox approach:
#   pod tars /workspace/output → uploads to litterbox.catbox.moe → we wget it locally
#
# Usage:  bash spikes/flux_lora_training/fetch_results.sh
# After fetching, STOP the pod on RunPod to halt billing.
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

OUT="$SPIKE_DIR/output"
mkdir -p "$OUT"

echo "[1/4] Checking training status..."

# Write expect script
EXPECT_SCRIPT=$(mktemp /tmp/fetch_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"' EXIT

cat > "$EXPECT_SCRIPT" << 'TCEOF'
#!/usr/bin/expect -f
set timeout 120

set key  [lindex $argv 0]
set host [lindex $argv 1]

proc ok {marker} {
    expect {
        $marker {}
        timeout { puts stderr "TIMEOUT: $marker"; exit 1 }
    }
    expect -re {#\s}
}

spawn ssh -t \
    -i $key \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o ServerAliveInterval=30 \
    $host

expect -re {#\s}

# Check tmux and list results
send "tmux has-session -t train 2>/dev/null && echo TMUX_ALIVE || echo TMUX_DONE; echo __S1__\r"
ok "__S1__"

send "echo SAFETENSORS:$(ls /workspace/output/gh0st_flux_lora_v1/*.safetensors 2>/dev/null | wc -l); echo __S2__\r"
ok "__S2__"

send "ls /workspace/output/gh0st_flux_lora_v1/ 2>/dev/null | head -20; echo __S3__\r"
ok "__S3__"

# Tar results and upload to litterbox.catbox.moe (24h temp hosting)
send "cd /workspace && tar -czf /tmp/lora_results.tar.gz output/ training.log 2>/dev/null; echo TAR_OK\r"
ok "TAR_OK"

send "ls -lh /tmp/lora_results.tar.gz; echo SIZE_OK\r"
ok "SIZE_OK"

send "curl -s -F 'reqtype=fileupload' -F 'time=24h' -F 'fileToUpload=@/tmp/lora_results.tar.gz' https://litterbox.catbox.moe/resources/internals/api.php ; echo UPLOAD_DONE\r"
set timeout 300
ok "UPLOAD_DONE"
set timeout 120

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

echo "[2/4] Fetching from pod (tar → litterbox.catbox.moe)..."
RAW=$(expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" 2>&1)

# Extract the catbox URL from output
URL=$(echo "$RAW" | grep -oE 'https://litter\.catbox\.moe/[a-zA-Z0-9]+\.(gz|tar\.gz|tgz)' | head -1)

echo "Raw output (last 20 lines):"
echo "$RAW" | tail -20

if [[ -z "$URL" ]]; then
  echo ""
  echo "[ERROR] Could not find catbox URL in output above."
  echo "The tar might still be uploading, or the session timed out."
  echo "Re-run this script when training completes."
  exit 1
fi

echo ""
echo "[3/4] Downloading from: $URL"
wget -q --show-progress "$URL" -O /tmp/lora_results.tar.gz

echo "[4/4] Extracting to $OUT/ ..."
tar -xzf /tmp/lora_results.tar.gz -C "$OUT" || tar -xzf /tmp/lora_results.tar.gz -C "$OUT" --strip-components=1

echo ""
echo "=== Local artifacts ==="
find "$OUT" -type f \( -name '*.safetensors' -o -name '*.png' -o -name 'training.log' \) | sort | head -40

echo ""
echo "=== DONE ==="
echo "Remember to STOP THE POD on RunPod to halt billing:"
echo "  https://www.runpod.io/console/pods"
echo "  Pod: $POD_HOST"
