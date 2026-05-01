#!/usr/bin/env bash
# Upload generate.py to the pod and run it in a tmux session.
# Pod must be running. Filesystem from training is preserved (dataset + LoRA already on pod).
# Usage: bash spikes/flux_lora_training/launch_generate.sh
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"
SCRIPT="$SPIKE_DIR/generate.py"

EXPECT_SCRIPT=$(mktemp /tmp/launch_gen_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"' EXIT

# Base64-encode the script to avoid quoting issues over PTY
SCRIPT_B64=$(base64 < "$SCRIPT")

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 120
set key  [lindex \$argv 0]
set host [lindex \$argv 1]
set b64  [lindex \$argv 2]

spawn ssh -t \\
    -i \$key \\
    -o StrictHostKeyChecking=no \\
    -o UserKnownHostsFile=/dev/null \\
    -o ServerAliveInterval=30 \\
    \$host

expect -re {#\s}

# Decode and write generate.py
send "echo '\$b64' | base64 -d > /workspace/generate.py && echo SCRIPT_OK\r"
expect {
    "SCRIPT_OK" {}
    timeout { puts "timeout writing script"; exit 1 }
}
expect -re {#\s}

# Verify dataset and LoRA are present
send "ls /workspace/dataset/lora_training_v2/ | head -5 && ls /workspace/output/gh0st_flux_lora_v2/gh0st_flux_lora_v2.safetensors && echo PREREQS_OK\r"
expect {
    "PREREQS_OK" {}
    timeout { puts "timeout checking prereqs"; exit 1 }
}
expect -re {#\s}

# Kill any old generate session, start fresh
send "tmux kill-session -t generate 2>/dev/null || true; echo KILL_OK\r"
expect {
    "KILL_OK" {}
    timeout {}
}
expect -re {#\s}

# Launch in tmux
send "tmux new-session -d -s generate 'cd /workspace && python generate.py 2>&1 | tee /workspace/generate.log'; echo LAUNCH_OK\r"
expect {
    "LAUNCH_OK" {}
    timeout { puts "timeout launching tmux"; exit 1 }
}
expect -re {#\s}

send "sleep 3 && tail -5 /workspace/generate.log; echo TAIL_OK\r"
expect {
    "TAIL_OK" {}
    timeout {}
}
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

echo "Uploading generate.py and launching on pod..."
expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$SCRIPT_B64" 2>&1 | \
  grep -v '^\[?2004' | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | \
  sed 's/\r//' | \
  grep -v '^$' | \
  grep -v '^spawn ' | \
  grep -v 'Warning:.*known_hosts' | \
  grep -v 'RUNPOD.IO' | \
  grep -v 'Enjoy your Pod'

echo ""
echo "=== Generation launched ==="
echo "Monitor with: bash spikes/flux_lora_training/monitor_generate.sh"
