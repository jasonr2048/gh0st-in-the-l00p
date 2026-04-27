#!/usr/bin/env bash
# Check training status on pod.
# RunPod gateway requires PTY — this script uses expect for the connection.
# Usage:
#   bash spikes/flux_lora_training/monitor.sh            # last 30 log lines + GPU
#   bash spikes/flux_lora_training/monitor.sh --status   # same as above
#   bash spikes/flux_lora_training/monitor.sh --last 80  # last N log lines
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

mode="${1:---status}"
n="${2:-30}"

# Write a temp expect script
EXPECT_SCRIPT=$(mktemp /tmp/monitor_XXXXXX.expect)
trap 'rm -f "$EXPECT_SCRIPT"' EXIT

cat > "$EXPECT_SCRIPT" << TCEOF
#!/usr/bin/expect -f
set timeout 60
set key  [lindex \$argv 0]
set host [lindex \$argv 1]
set n    [lindex \$argv 2]

spawn ssh -t \\
    -i \$key \\
    -o StrictHostKeyChecking=no \\
    -o UserKnownHostsFile=/dev/null \\
    -o ServerAliveInterval=30 \\
    \$host

expect -re {#\s}

send "tmux has-session -t train 2>/dev/null && echo TMUX:running || echo TMUX:stopped; echo __T1__\r"
expect {
    "__T1__" {}
    timeout { puts "timeout checking tmux"; exit 1 }
}
expect -re {#\s}

send "echo '=== last \$n log lines ==='; tail -n \$n /workspace/training.log 2>/dev/null || echo '(no log yet)'; echo __T2__\r"
expect {
    "__T2__" {}
    timeout { puts "timeout reading log"; exit 1 }
}
expect -re {#\s}

send "echo '=== GPU ==='; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; echo __T3__\r"
expect {
    "__T3__" {}
    timeout { puts "timeout checking GPU" }
}
expect -re {#\s}

send "echo '=== samples ==='; find /workspace/output -name '*.png' 2>/dev/null | sort | tail -10; echo __T4__\r"
expect {
    "__T4__" {}
    timeout {}
}
expect -re {#\s}

send "exit\r"
expect eof
TCEOF
chmod +x "$EXPECT_SCRIPT"

expect "$EXPECT_SCRIPT" "$KEY" "$POD_HOST" "$n" 2>&1 | \
  grep -v '^\[?2004' | \
  sed 's/\x1b\[[0-9;]*[mKHF]//g' | \
  sed 's/\r//' | \
  grep -v '^$' | \
  grep -v '^spawn ' | \
  grep -v 'Warning:.*known_hosts' | \
  grep -v 'RUNPOD.IO' | \
  grep -v 'Enjoy your Pod' | \
  grep -v '^--$' | \
  grep -v '__T[1-4]__' | \
  grep -v 'exit$' | \
  grep -v 'Connection to.*closed'
