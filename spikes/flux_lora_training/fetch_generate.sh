#!/usr/bin/env bash
# Fetch generated images from pod → local output/gh0st_flux_lora_v2/generated/
# Uses tar → litterbox.catbox.moe → local wget (RunPod blocks rsync/scp).
# Usage: bash spikes/flux_lora_training/fetch_generate.sh
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

OUT="$SPIKE_DIR/output/gh0st_flux_lora_v2/generated"
mkdir -p "$OUT"

echo "=== [1/4] Connecting to pod, taring generated output... ==="

EXPECT_SCRIPT=$(mktemp /tmp/fetch_gen_XXXXXX.expect)
URL_FILE=$(mktemp /tmp/fetch_gen_url_XXXXXX.txt)
trap 'rm -f "$EXPECT_SCRIPT" "$URL_FILE"' EXIT

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

# Show what we have
send "echo 'clean:' && find /workspace/output/gh0st_flux_lora_v2/generated/clean -name '*.png' 2>/dev/null | wc -l && echo 'mashups:' && find /workspace/output/gh0st_flux_lora_v2/generated/mashups -name '*.png' 2>/dev/null | wc -l && echo 'grids:' && find /workspace/output/gh0st_flux_lora_v2/generated -name '*_grid.jpg' 2>/dev/null | wc -l; echo __S1__\r"
ok "__S1__"

# Tar only generated dir (skip LoRA weights to keep size down)
send "cd /workspace/output/gh0st_flux_lora_v2 && tar -czf /tmp/generated.tar.gz generated/ && ls -lh /tmp/generated.tar.gz; echo TAR_OK\r"
set timeout 120
ok "TAR_OK"

# Upload to catbox
send "curl -s -F 'reqtype=fileupload' -F 'time=24h' -F 'fileToUpload=@/tmp/generated.tar.gz' https://litterbox.catbox.moe/resources/internals/api.php; echo UPLOAD_DONE\r"
set timeout 600
ok "UPLOAD_DONE"

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

echo "=== [2/4] Uploading tar to litterbox.catbox.moe... ==="
RAW=$(expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" 2>&1 | \
  grep -v '^\[?2004' | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | \
  sed 's/\r//')

echo "$RAW" | tail -30

URL=$(echo "$RAW" | grep -oE 'https://litter\.catbox\.moe/[a-zA-Z0-9]+\.(gz|tar\.gz|tgz)' | head -1)

if [[ -z "$URL" ]]; then
  echo ""
  echo "[ERROR] Could not find catbox URL in output."
  exit 1
fi

echo ""
echo "=== [3/4] Downloading from: $URL ==="
wget -q --show-progress "$URL" -O /tmp/generated.tar.gz

echo ""
echo "=== [4/4] Extracting to $OUT/... ==="
tar -xzf /tmp/generated.tar.gz -C "$SPIKE_DIR/output/gh0st_flux_lora_v2/" || \
  tar -xzf /tmp/generated.tar.gz -C "$SPIKE_DIR/output/gh0st_flux_lora_v2/" --strip-components=1

echo ""
echo "=== Local output ==="
echo "Clean images:  $(find "$OUT/clean" -name '*.png' 2>/dev/null | wc -l)"
echo "Mashup images: $(find "$OUT/mashups" -name '*.png' 2>/dev/null | wc -l)"
echo "Grid files:    $(find "$OUT" -name '*_grid.jpg' 2>/dev/null | wc -l)"
echo ""
echo "=== DONE ==="
echo "IMPORTANT: Stop the pod to halt billing:"
echo "  https://www.runpod.io/console/pods"
