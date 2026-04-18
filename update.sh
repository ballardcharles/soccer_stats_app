#!/bin/bash
# update.sh
# ---------
# Full data refresh: collect → build → done.
#
# Run manually any time:
#     ./update.sh
#
# Or install the launchd schedule (runs automatically Mon/Wed/Fri at 07:00):
#     cp com.soccerstats.update.plist ~/Library/LaunchAgents/
#     launchctl load ~/Library/LaunchAgents/com.soccerstats.update.plist
#
# Logs are written to logs/update.log (last 500 lines kept).

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/soccer_env/bin/python"
LOG_FILE="$SCRIPT_DIR/logs/update.log"

cd "$SCRIPT_DIR"
mkdir -p logs

# ── Logging helper ────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$LOG_FILE"
}

# ── Start ─────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════"
log "Soccer Stats — weekly update starting"
log "════════════════════════════════════════"

# ── Step 1: Collect new match data ────────────────────────────────────────────
log "Step 1/3 — Running collectors (Understat · ESPN · WhoScored)…"
if "$PYTHON" run_collection.py >> "$LOG_FILE" 2>&1; then
    log "Step 1/3 — Collection complete ✓"
else
    log "Step 1/3 — Collection FAILED (exit $?). Check logs/update.log."
    exit 1
fi

# ── Step 2: Rebuild processed data + soccer_stats.db ─────────────────────────
log "Step 2/3 — Rebuilding processed CSVs and soccer_stats.db…"
if "$PYTHON" build_processed.py >> "$LOG_FILE" 2>&1; then
    log "Step 2/3 — Build complete ✓"
else
    log "Step 2/3 — Build FAILED (exit $?). Check logs/update.log."
    exit 1
fi

# ── Step 3: Push updated database to GitHub ───────────────────────────────────
log "Step 3/3 — Pushing updated soccer_stats.db to GitHub…"
if git -C "$SCRIPT_DIR" diff --quiet HEAD -- soccer_stats.db 2>/dev/null; then
    log "Step 3/3 — soccer_stats.db unchanged, skipping push."
else
    git -C "$SCRIPT_DIR" add soccer_stats.db >> "$LOG_FILE" 2>&1
    git -C "$SCRIPT_DIR" commit -m "Auto-update: soccer_stats.db $(date '+%Y-%m-%d')" >> "$LOG_FILE" 2>&1
    if git -C "$SCRIPT_DIR" push >> "$LOG_FILE" 2>&1; then
        log "Step 3/3 — Push complete ✓  (Streamlit Cloud will redeploy automatically)"
    else
        log "Step 3/3 — Push FAILED. Check git credentials or LFS quota."
        exit 1
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
log "Update finished successfully."
log ""

# Keep log file from growing unbounded — trim to last 500 lines
tail -500 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
