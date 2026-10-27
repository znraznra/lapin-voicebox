#!/usr/bin/env bash
# Start Voicebox from a source checkout (backend + web UI) and open it in the browser.
# Installed as an app-menu entry by scripts/linux/install-desktop-entry.sh.
# Logs: ~/.local/state/voicebox/launcher.log
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_URL="http://127.0.0.1:17493"
WEB_PORT=5173
WEB_URL="http://localhost:${WEB_PORT}"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/voicebox"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/launcher.log" 2>&1
echo "=== $(date) starting from $REPO"

notify() { command -v notify-send >/dev/null 2>&1 && notify-send "Voicebox" "$1" || true; }
die() { echo "ERROR: $1"; notify "$1 (see $LOG_DIR/launcher.log)"; exit 1; }

[ -x "$REPO/backend/venv/bin/uvicorn" ] || die "No Python environment. Run 'just setup' in $REPO first."
command -v bun >/dev/null 2>&1 || die "bun not found on PATH."

# If it is already up, just open it.
if curl -sf "$WEB_URL" >/dev/null 2>&1 && curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
    echo "already running"
    xdg-open "$WEB_URL" >/dev/null 2>&1 || true
    exit 0
fi

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "$REPO"
if ! curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
    echo "starting backend"
    backend/venv/bin/uvicorn backend.main:app --port 17493 &
    pids+=($!)
fi
echo "starting web UI"
(cd web && exec bun run dev --port "$WEB_PORT" --strictPort) &
pids+=($!)

# Wait up to ~90s for both, then open the browser.
for _ in $(seq 1 90); do
    if curl -sf "$WEB_URL" >/dev/null 2>&1 && curl -sf "$BACKEND_URL/health" >/dev/null 2>&1; then
        echo "ready"
        xdg-open "$WEB_URL" >/dev/null 2>&1 || true
        break
    fi
    sleep 1
done
curl -sf "$WEB_URL" >/dev/null 2>&1 || die "Voicebox did not start within 90s."

# Stay alive while the servers run; closing the launcher (or killing it) stops them.
wait -n "${pids[@]}" || true
echo "a server exited; shutting down"
