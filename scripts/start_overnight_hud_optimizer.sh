#!/usr/bin/env sh
set -eu

# Start an unattended Protean optimizer run that streams every trial to HUD.
# Configure with environment variables; no secrets are stored in this file.

DURATION_HOURS="${DURATION_HOURS:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-1000000}"
POLICY="${POLICY:-fireworks}"
HUD_GROUP="${HUD_GROUP:-1}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-runs/protean-overnight-${STAMP}}"
HUD_JOB_NAME="${HUD_JOB_NAME:-protean-overnight-${STAMP}}"
FIREWORKS_MODEL_ARG="${FIREWORKS_MODEL:+--fireworks-model ${FIREWORKS_MODEL}}"
CONTROLLER_ARG="${CONTROLLER:+--controller ${CONTROLLER}}"

if [ -z "${HUD_API_KEY:-}" ] && [ -f "$HOME/.hud/.env" ]; then
  HUD_API_KEY="$(awk -F= '/^HUD_API_KEY=/{gsub(/"/,"",$2); gsub(/\047/,"",$2); print $2}' "$HOME/.hud/.env" | tail -n 1)"
  export HUD_API_KEY
fi

if [ -z "${HUD_API_KEY:-}" ]; then
  echo "HUD_API_KEY is required. Run 'hud set HUD_API_KEY=...' or export HUD_API_KEY." >&2
  exit 2
fi

if [ "$POLICY" = "fireworks" ] && [ -z "${FIREWORKS_API_KEY:-}" ]; then
  echo "FIREWORKS_API_KEY is required when POLICY=fireworks." >&2
  exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

LOG_PATH="$OUT_DIR/overnight.log"
PID_PATH="$OUT_DIR/pid"

# shellcheck disable=SC2086
nohup "$PYTHON_BIN" scripts/run_optimizer.py \
  --edit-policy "$POLICY" \
  --all-ops \
  --duration-hours "$DURATION_HOURS" \
  --max-rounds "$MAX_ROUNDS" \
  --stream-hud \
  --hud-group "$HUD_GROUP" \
  --hud-job-name "$HUD_JOB_NAME" \
  --out-dir "$OUT_DIR" \
  $FIREWORKS_MODEL_ARG \
  $CONTROLLER_ARG \
  > "$LOG_PATH" 2>&1 &

PID="$!"
echo "$PID" > "$PID_PATH"

cat <<EOF
Started Protean overnight optimizer.
  pid:        $PID
  duration:   ${DURATION_HOURS}h
  policy:     $POLICY
  hud job:    $HUD_JOB_NAME
  out dir:    $OUT_DIR
  log:        $LOG_PATH

Watch it:
  tail -f $LOG_PATH
  find $OUT_DIR -name 'improvements_*.jsonl' -maxdepth 2 -print
EOF
