#!/usr/bin/env bash
# stack_sessions.sh — Stack OME-TIFF frames for all session dirs matching prefix
# Usage: stack_sessions.sh --prefix stroh-ej [--dry-run]

set -euo pipefail

PREFIX=""
DRY_RUN=false
SCRIPT_DIR=""

usage() {
    echo "Usage: $(basename "$0") --prefix <lab>-<experimenter> [--script-dir <path>] [--dry-run]"
    echo ""
    echo "  --prefix      Lab/experimenter prefix, e.g. stroh-ej"
    echo "  --script-dir  Directory containing stack_to_bigtiff.py (default: same dir as this script)"
    echo "  --dry-run     Preview without writing any files"
    exit 1
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)     PREFIX="$2";     shift 2 ;;
        --script-dir) SCRIPT_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift   ;;
        -h|--help)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$PREFIX" ]] && { echo "Error: --prefix is required"; usage; }

# --- Locate stack_to_bigtiff.py ---
if [[ -z "$SCRIPT_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
STACKER="$SCRIPT_DIR/stack_to_bigtiff.py"

if [[ ! -f "$STACKER" ]]; then
    echo "Error: stack_to_bigtiff.py not found at $STACKER"
    exit 1
fi

# --- Main loop ---
shopt -s nullglob
session_dirs=( "${PREFIX}"-*/ )

if [[ ${#session_dirs[@]} -eq 0 ]]; then
    echo "No session directories found matching: ${PREFIX}-*/"
    exit 1
fi

echo "Prefix      : $PREFIX"
echo "Script dir  : $SCRIPT_DIR"
echo "Sessions    : ${#session_dirs[@]}"
echo "Dry run     : $DRY_RUN"
echo ""

for session_dir in "${session_dirs[@]}"; do
    session="${session_dir%/}"
    echo "==> $session"

    # Discover channels present in this session
    channels=$(ls "${session_dir}"*_C[0-9][0-9]_t*.ome.tif 2>/dev/null \
        | grep -oP '_C\K[0-9]+' | sort -u)

    if [[ -z "$channels" ]]; then
        echo "    No OME-TIFF frames found, skipping"
        continue
    fi

    for ch in $channels; do
        output="${session_dir}${session}_C${ch}.ome.tif"

        if [[ -f "$output" ]]; then
            echo "    C${ch}: already exists, skipping"
            continue
        fi

        frames=( "${session_dir}"*_C${ch}_t*.ome.tif )
        echo "    C${ch}: ${#frames[@]} frames -> $output"

        if $DRY_RUN; then
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.ome.tif" \
                --output "$output" \
                --dry-run
        else
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.ome.tif" \
                --output "$output"
        fi
    done

    echo ""
done
