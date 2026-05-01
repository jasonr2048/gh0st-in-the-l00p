#!/usr/bin/env bash
# Fetch sequence images from pod → local output/gh0st_flux_lora_v2/sequence/
# Uses tar → litterbox.catbox.moe → local wget.
# Usage: bash spikes/flux_lora_training/fetch_sequence.sh
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

OUT_BASE="$SPIKE_DIR/output/gh0st_flux_lora_v2"
mkdir -p "$OUT_BASE"

echo "=== [1/4] Connecting to pod... ==="

EXPECT_SCRIPT=$(mktemp /tmp/fetch_seq_XXXXXX.expect)
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

send "echo 'pure:' && find /workspace/output/gh0st_flux_lora_v2/sequence/pure -name '*.png' 2>/dev/null | wc -l && find /workspace/output/gh0st_flux_lora_v2/sequence -name '*.png' 2>/dev/null | wc -l && echo COUNTS_OK\r"
ok "COUNTS_OK"

send "cd /workspace/output/gh0st_flux_lora_v2 && tar -czf /tmp/sequence.tar.gz sequence/ && ls -lh /tmp/sequence.tar.gz; echo TAR_OK\r"
set timeout 120
ok "TAR_OK"

send "curl -s -F 'reqtype=fileupload' -F 'time=24h' -F 'fileToUpload=@/tmp/sequence.tar.gz' https://litterbox.catbox.moe/resources/internals/api.php; echo UPLOAD_DONE\r"
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

echo "$RAW" | tail -15

URL=$(echo "$RAW" | grep -oE 'https://litter\.catbox\.moe/[a-zA-Z0-9]+\.(gz|tar\.gz|tgz)' | head -1)

if [[ -z "$URL" ]]; then
  echo ""
  echo "[ERROR] No catbox URL found. Re-run when generation completes."
  exit 1
fi

echo ""
echo "=== [3/4] Downloading from: $URL ==="
wget -q --show-progress "$URL" -O /tmp/sequence.tar.gz

echo ""
echo "=== [4/4] Extracting to $OUT_BASE/ ==="
tar -xzf /tmp/sequence.tar.gz -C "$OUT_BASE/"

echo ""
echo "=== Local output ==="
echo "Pure frames : $(find "$OUT_BASE/sequence/pure" -name '*.png' 2>/dev/null | wc -l)"
for d in "$OUT_BASE"/sequence/*/; do
  name=$(basename "$d")
  [[ "$name" == "pure" ]] && continue
  count=$(find "$d" -name '*.png' 2>/dev/null | wc -l)
  echo "$name: $count"
done
echo ""
echo "=== DONE ==="
echo "Stop the pod: https://www.runpod.io/console/pods"
