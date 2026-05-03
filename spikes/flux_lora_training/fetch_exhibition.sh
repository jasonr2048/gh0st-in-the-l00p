#!/usr/bin/env bash
# Fetch exhibition stills from pod → local output/.
# RunPod's SSH proxy forces PTY so direct tar-pipe/rsync don't work.
# Workaround: tar on pod → upload to catbox → wget locally.
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

echo "=== [1/3] Tarring + uploading from pod to catbox ==="

EXPECT_SCRIPT=$(mktemp /tmp/fetch_exh_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 60
set key  [lindex \$argv 0]
set host [lindex \$argv 1]
set name [lindex \$argv 2]

spawn ssh -t \\
    -i \$key \\
    -o StrictHostKeyChecking=no \\
    -o UserKnownHostsFile=/dev/null \\
    -o ServerAliveInterval=30 \\
    \$host

expect -re {#\s}

# Count what's there
send "find /workspace/output/\$name -name '*.png' | wc -l; echo COUNT_OK\r"
expect { "COUNT_OK" {} timeout { puts "timeout COUNT"; exit 1 } }
expect -re {#\s}

# Tar it
send "cd /workspace/output && tar -czf /tmp/\${name}.tar.gz \$name/ && ls -lh /tmp/\${name}.tar.gz; echo TAR_OK\r"
set timeout 120
expect { "TAR_OK" {} timeout { puts "timeout TAR"; exit 1 } }
expect -re {#\s}

# Upload to catbox — print URL on its own line then a marker
send "echo CATBOX_START && curl -s -F 'reqtype=fileupload' -F 'time=24h' -F 'fileToUpload=@/tmp/\${name}.tar.gz' https://litterbox.catbox.moe/resources/internals/api.php && echo CATBOX_END\r"
set timeout 600
expect { "CATBOX_END" {} timeout { puts "timeout CATBOX"; exit 1 } }
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

RAW=$(expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$OUTPUT_NAME" 2>&1 | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | tr -d '\r')

# Print filtered output for visibility
echo "$RAW" | grep -Ev '^\[.2004|RUNPOD|Enjoy|Warning|Permanently|^spawn ' | \
  grep -v '^$' | tail -20

# Extract URL — it appears between CATBOX_START and CATBOX_END
URL=$(echo "$RAW" | grep -oE 'https://litter\.catbox\.moe/[a-zA-Z0-9.]+' | head -1)

if [[ -z "$URL" ]]; then
    echo ""
    echo "[ERROR] No catbox URL found. Raw output above — check for upload errors."
    exit 1
fi

echo ""
echo "=== [2/3] Downloading from catbox: $URL ==="
wget -q --show-progress "$URL" -O "/tmp/${OUTPUT_NAME}.tar.gz"

echo ""
echo "=== [3/3] Extracting to $LOCAL_OUT/ ==="
tar -xzf "/tmp/${OUTPUT_NAME}.tar.gz" -C "$SPIKE_DIR/output/"
rm -f "/tmp/${OUTPUT_NAME}.tar.gz"

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
echo "=== DONE — stop the pod: https://www.runpod.io/console/pods ==="
