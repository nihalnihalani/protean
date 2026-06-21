#!/usr/bin/env sh
set -eu

OUT_DIR="${1:-runs/protean-overnight}"
PID_PATH="$OUT_DIR/pid"
LOG_PATH="$OUT_DIR/overnight.log"

if [ ! -d "$OUT_DIR" ]; then
  echo "Run directory does not exist: $OUT_DIR" >&2
  exit 2
fi

if [ -f "$PID_PATH" ]; then
  PID="$(cat "$PID_PATH")"
  if ps -p "$PID" >/dev/null 2>&1; then
    echo "status: running"
  else
    echo "status: stopped"
  fi
  echo "pid: $PID"
else
  echo "status: unknown"
  echo "pid: not recorded"
fi

echo "out_dir: $OUT_DIR"
echo "log: $LOG_PATH"

if [ -f "$LOG_PATH" ]; then
  echo
  echo "last log lines:"
  tail -n 20 "$LOG_PATH"
fi

echo
echo "latest improvement rows:"
find "$OUT_DIR" -name 'improvements_*.jsonl' -maxdepth 3 -print | while IFS= read -r path; do
  echo "==> $path"
  tail -n 3 "$path"
done
