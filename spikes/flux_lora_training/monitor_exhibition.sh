#!/usr/bin/env bash
# Check exhibition generation progress on pod.
# Usage: bash spikes/flux_lora_training/monitor_exhibition.sh [--last N]
set -euo pipefail

SPIKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SPIKE_DIR/.runpod_env"
KEY="$SPIKE_DIR/.runpod_key"

n=40
while [[ $# -gt 0 ]]; do
    case "$1" in
        --last) n="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EXPECT_SCRIPT=$(mktemp /tmp/mon_exh_XXXXXX.expect)
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

send "tmux has-session -t exhibition 2>/dev/null && echo 'STATUS: running' || echo 'STATUS: stopped'; echo __M1__\r"
expect { "__M1__" {} timeout { puts "timeout M1"; exit 1 } }
expect -re {#\s}

send "echo '=== last \$n log lines ===' && tail -n \$n /workspace/exhibition.log 2>/dev/null || echo '(no log yet)'; echo __M2__\r"
expect { "__M2__" {} timeout { puts "timeout M2"; exit 1 } }
expect -re {#\s}

send "echo '=== output counts ===' && for d in /workspace/output/gh0st_exhibition_*/; do echo \"--- \$(basename \$d)\"; find \"\$d\" -name '*.png' | wc -l; done 2>/dev/null || echo '(no output yet)'; echo __M3__\r"
expect { "__M3__" {} timeout {} }
expect -re {#\s}

send "echo '=== GPU ===' && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; echo __M4__\r"
expect { "__M4__" {} timeout {} }
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
  grep -v '__M[1-4]__' | \
  grep -v 'exit$' | \
  grep -v 'Connection to.*closed'
