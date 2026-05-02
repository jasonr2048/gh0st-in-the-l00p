#!/usr/bin/env bash
# Fetch exhibition stills from pod → local output/
# Uses tar → litterbox.catbox.moe → local wget.
#
# Usage:
#   bash spikes/flux_lora_training/fetch_exhibition.sh
#   bash spikes/flux_lora_training/fetch_exhibition.sh --output gh0st_exhibition_v2
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

OUTPUT_NAME="gh0st_exhibition_v1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

LOCAL_OUT="$SPIKE_DIR/output/$OUTPUT_NAME"
mkdir -p "$LOCAL_OUT"

POD_OUT="/workspace/output/$OUTPUT_NAME"

echo "=== [1/4] Connecting to pod to tar & upload... ==="

EXPECT_SCRIPT=$(mktemp /tmp/fetch_exh_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 300
set key      [lindex \$argv 0]
set host     [lindex \$argv 1]
set pod_out  [lindex \$argv 2]

proc ok {marker} {
    expect {
        \$marker {}
        timeout { puts stderr "TIMEOUT: \$marker"; exit 1 }
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

# Count what's there
send "echo '=== output counts ===' && find \$pod_out -name '*.png' 2>/dev/null | wc -l && ls \$pod_out 2>/dev/null; echo COUNTS_OK\r"
ok "COUNTS_OK"

# Tar the output
send "cd /workspace/output && tar -czf /tmp/exhibition_out.tar.gz \$(basename \$pod_out)/ && ls -lh /tmp/exhibition_out.tar.gz; echo TAR_OK\r"
set timeout 300
ok "TAR_OK"

# Upload to catbox
send "curl -s -F 'reqtype=fileupload' -F 'time=24h' -F 'fileToUpload=@/tmp/exhibition_out.tar.gz' https://litterbox.catbox.moe/resources/internals/api.php; echo UPLOAD_DONE\r"
set timeout 600
ok "UPLOAD_DONE"

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

echo "=== [2/4] Uploading tar to litterbox.catbox.moe... ==="
RAW=$(expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$POD_OUT" 2>&1 | \
  grep -v '^\[?2004' | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | \
  sed 's/\r//')

echo "$RAW" | tail -20

URL=$(echo "$RAW" | grep -oE 'https://litter\.catbox\.moe/[a-zA-Z0-9]+\.(gz|tar\.gz|tgz)' | head -1)

if [[ -z "$URL" ]]; then
    echo ""
    echo "[ERROR] No catbox URL found in output above."
    echo "  The generation may still be running — check with monitor_exhibition.sh"
    exit 1
fi

echo ""
echo "=== [3/4] Downloading from: $URL ==="
wget -q --show-progress "$URL" -O /tmp/exhibition_out.tar.gz

echo ""
echo "=== [4/4] Extracting to $LOCAL_OUT/ ==="
tar -xzf /tmp/exhibition_out.tar.gz -C "$SPIKE_DIR/output/"

echo ""
echo "=== Local output: $LOCAL_OUT ==="
for cat_dir in "$LOCAL_OUT"/*/; do
    name=$(basename "$cat_dir")
    count=$(find "$cat_dir" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
    echo "  $name: $count stills"
done
total=$(find "$LOCAL_OUT" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
echo "  Total: $total stills"
echo ""
echo "=== DONE ==="
echo "Stop the pod when fetching is confirmed complete:"
echo "  https://www.runpod.io/console/pods"
