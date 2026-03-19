#!/bin/bash
# swap_watch.sh — poll swap per process every N seconds while pipeline runs.
# Usage: bash swap_watch.sh [interval_seconds=30]
#
# Run this in a separate terminal alongside the pipeline.
# It logs a condensed swap snapshot every interval seconds.
# Ctrl+C to stop.

INTERVAL=${1:-30}
OUTFILE="swap_watch_$(date +%H%M%S).log"

echo "Logging to $OUTFILE every ${INTERVAL}s. Ctrl+C to stop."
echo "Timestamp            PID       RSS_MB  Swap_MB  CMD" | tee "$OUTFILE"
echo "─────────────────────────────────────────────────────────────" | tee -a "$OUTFILE"

while true; do
    TS=$(date '+%H:%M:%S')
    # Get all python processes with non-zero swap
    while IFS= read -r pid; do
        [ -f "/proc/$pid/status" ] || continue
        CMD=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-40)
        RSS=$(awk '/^VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null)
        SWAP=$(awk '/^VmSwap/{print $2}' "/proc/$pid/status" 2>/dev/null)
        [ -z "$SWAP" ] && continue
        [ "$SWAP" -gt 1024 ] 2>/dev/null || continue
        RSS_MB=$(( ${RSS:-0} / 1024 ))
        SWAP_MB=$(( SWAP / 1024 ))
        printf "%-20s %-9s %7d  %7d  %s\n" "$TS" "$pid" "$RSS_MB" "$SWAP_MB" "$CMD" | tee -a "$OUTFILE"
    done < <(ls /proc | grep '^[0-9]')
    
    # System total
    TOTAL_SWAP=$(free -m | awk '/Swap/{print $3}')
    printf "%-20s %-9s %7s  %7d  [SYSTEM TOTAL]\n" "$TS" "---" "---" "$TOTAL_SWAP" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
    sleep "$INTERVAL"
done
