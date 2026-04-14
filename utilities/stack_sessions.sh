#!/usr/bin/env bash
# stack_sessions.sh — Stack OME-TIFF frames for all session dirs under a parent dir
#
# Works at the level of:
#   /data/src/stroh-ia/imaging/stroh-ia-STO0647-20150113/
#
# Output filename format:
#   stroh-ia-20150113-STO0647-TL001_122919-LP03p30-fa1-spont-25x-C00-fc010000-fs30p00.tif
#
# Usage:
#   stack_sessions.sh --prefix stroh-ia --parent /data/src/stroh-ia/imaging/stroh-ia-STO0647-20150113 [--script-dir <path>] [--dry-run]

set -euo pipefail

PREFIX=""
PARENT_DIR=""
DRY_RUN=false
SCRIPT_DIR=""

usage() {
    echo "Usage: $(basename "$0") --prefix <lab>-<experimenter> --parent <dir> [--script-dir <path>] [--dry-run]"
    echo ""
    echo "  --prefix      Session directory prefix, e.g. stroh-ej or stroh-ia"
    echo "  --parent      Parent directory containing session subdirectories"
    echo "  --script-dir  Directory containing stack_to_bigtiff.py and ome_meta.py"
    echo "                (default: same directory as this script)"
    echo "  --dry-run     Preview without writing any files"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)     PREFIX="$2";     shift 2 ;;
        --parent)     PARENT_DIR="$2"; shift 2 ;;
        --script-dir) SCRIPT_DIR="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift   ;;
        -h|--help)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$PREFIX" ]]       && { echo "Error: --prefix is required";                          usage; }
[[ -z "$PARENT_DIR" ]]   && { echo "Error: --parent is required";                          usage; }
[[ ! -d "$PARENT_DIR" ]] && { echo "Error: --parent is not a directory: $PARENT_DIR"; exit 1; }

if [[ -z "$SCRIPT_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

STACKER="$SCRIPT_DIR/stack_to_bigtiff.py"
OME_META="$SCRIPT_DIR/ome_meta.py"

[[ ! -f "$STACKER" ]]  && { echo "Error: stack_to_bigtiff.py not found at $STACKER"; exit 1; }
[[ ! -f "$OME_META" ]] && { echo "Error: ome_meta.py not found at $OME_META";         exit 1; }

# --- Main loop ---
shopt -s nullglob
session_dirs=( "${PARENT_DIR}/${PREFIX}"-*/ )

if [[ ${#session_dirs[@]} -eq 0 ]]; then
    echo "No session directories found matching: ${PARENT_DIR}/${PREFIX}-*/"
    exit 1
fi

echo "Parent dir  : $PARENT_DIR"
echo "Prefix      : $PREFIX"
echo "Script dir  : $SCRIPT_DIR"
echo "Sessions    : ${#session_dirs[@]}"
echo "Dry run     : $DRY_RUN"
echo ""

for session_dir in "${session_dirs[@]}"; do
    session="${session_dir%/}"
    session_name="$(basename "$session")"
    echo "==> $session_name"

    # Find the master OME-TIFF — must be an actual frame file (contains OME-XML
    # and real pixel data).  Companion/descriptor files from some acquisition
    # software (e.g. FluoView, ScanImage redirect stubs) share the .ome.tif
    # extension but are plain text and will fail tifffile.  Prefer *_t0000.ome.tif
    # (first time-point), which is always a real frame; fall back to any .ome.tif
    # only when no _t0000 files exist.
    master=""
    for _candidate in \
        "$(ls "${session_dir}"*_t0000.ome.tif 2>/dev/null | sort | head -1)" \
        "$(ls "${session_dir}"*.ome.tif        2>/dev/null | sort | head -1)"
    do
        [[ -z "$_candidate" || ! -f "$_candidate" ]] && continue
        # Quick magic-byte check: TIFF files start with 'II' (0x4949) or 'MM' (0x4d4d)
        _magic="$(head -c 2 "$_candidate" 2>/dev/null | od -A n -t x1 | tr -d ' \n')"
        if [[ "$_magic" == "4949" || "$_magic" == "4d4d" ]]; then
            master="$_candidate"
            break
        fi
    done
    if [[ -z "$master" ]]; then
        echo "    No OME-TIFF files found, skipping"
        continue
    fi

    # Read frameCount and samplingRate from OME-XML
    meta=$(python "$OME_META" "$master" 2>&1) || {
        echo "    ome_meta.py failed: $meta"
        continue
    }
    eval "$meta"   # sets $frameCount and $samplingRate

    echo "    Master      : $(basename "$master")"
    echo "    Frame count : $frameCount"
    echo "    Sample rate : ${samplingRate} Hz"

    # Discover channels
    channels=$(ls "${session_dir}"*_C[0-9][0-9]_t*.ome.tif 2>/dev/null \
        | grep -oP '_C\K[0-9]+' | sort -u || true)

    if [[ -z "$channels" ]]; then
        echo "    No channel frame files found, skipping"
        continue
    fi

    for ch in $channels; do
        output="${session_dir}${session_name}-C${ch}-fc${frameCount}.tif"

        if [[ -f "$output" ]]; then
            echo "    C${ch}: already exists, skipping"
            continue
        fi

        frames=( "${session_dir}"*_C${ch}_t*.ome.tif )
        echo "    C${ch}: ${#frames[@]} frames -> $(basename "$output")"

        if $DRY_RUN; then
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.ome.tif" \
                --output "$output" \
                --preallocate \
                --compression none \
                --dry-run
        else
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.ome.tif" \
                --output "$output" \
                --preallocate \
                --compression none

            # Update Trial.yaml with OME metadata (only once, on first channel)
            if [[ "$ch" == "$(echo "$channels" | head -1)" ]]; then
                yaml_path="${session_dir}${session_name}.yaml"
                if [[ -f "$yaml_path" ]]; then
                    echo "    yaml: updating $yaml_path with OME metadata"
                    python "$OME_META" "$master" --update-yaml "$yaml_path" || echo "    yaml: WARNING ome_meta update failed"
                else
                    echo "    yaml: $yaml_path not found, skipping OME update"
                fi
            fi
        fi
    done

    echo ""
done
