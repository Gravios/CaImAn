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
#   stack_sessions.sh --prefix stroh-ia --parent /data/src/stroh-ia/imaging/stroh-ia-STO0647-20150113 \
#                     [--script-dir <path>] [--delete-sources] [--dry-run]

set -euo pipefail

PREFIX=""
PARENT_DIR=""
DRY_RUN=false
SCRIPT_DIR=""
DELETE_SOURCES=false

usage() {
    echo "Usage: $(basename "$0") --prefix <lab>-<experimenter> --parent <dir> [options]"
    echo ""
    echo "  --prefix          Session directory prefix, e.g. stroh-ej or stroh-ia"
    echo "  --parent          Parent directory containing session subdirectories"
    echo "  --script-dir      Directory containing stack_to_bigtiff.py and ome_meta.py"
    echo "                    (default: same directory as this script)"
    echo "  --delete-sources  Delete source frame TIFFs after each successful stack write."
    echo "                    Off by default — source frames are preserved unless this flag"
    echo "                    is passed explicitly."
    echo "  --dry-run         Preview without writing any files"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)          PREFIX="$2";     shift 2 ;;
        --parent)          PARENT_DIR="$2"; shift 2 ;;
        --script-dir)      SCRIPT_DIR="$2"; shift 2 ;;
        --delete-sources)  DELETE_SOURCES=true; shift ;;
        --dry-run)         DRY_RUN=true;    shift   ;;
        -h|--help)         usage ;;
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
echo "Del sources : $DELETE_SOURCES"
echo ""

for session_dir in "${session_dirs[@]}"; do
    session="${session_dir%/}"
    session_name="$(basename "$session")"
    echo "==> $session_name"

    # Find the master OME-TIFF — any frame that carries valid OME-XML.
    # Strategy: try the earliest-sorted _t0* file (handles both unpadded _t0000
    # and already-padded _t000000); then fall back to any .ome.tif that passes
    # the TIFF magic-byte check.  This is resilient to partial deletions that
    # may have removed the original t0000 carrier.
    master=""
    for _candidate in \
        "$(ls "${session_dir}"*_t0*.ome.tif 2>/dev/null | sort | head -1)" \
        "$(ls "${session_dir}"*.ome.tif      2>/dev/null | sort | head -1)"
    do
        [[ -z "$_candidate" || ! -f "$_candidate" ]] && continue
        _magic="$(head -c 2 "$_candidate" 2>/dev/null | od -A n -t x1 | tr -d ' \n')"
        if [[ "$_magic" == "4949" || "$_magic" == "4d4d" ]]; then
            # Verify it actually carries OME-XML before accepting it
            if python "$OME_META" "$_candidate" >/dev/null 2>&1; then
                master="$_candidate"
                break
            fi
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

    # Discover channels — match both unpadded (_t0000) and padded (_t000000) names
    channel_ids=()
    for _f in "${session_dir}"*_C[0-9]*_t0*.ome.tif; do
        [[ -f "$_f" ]] || continue
        _bn="$(basename "$_f")"
        _cid="${_bn##*_C}"
        _cid="${_cid%%_*}"
        [[ "$_cid" =~ ^[0-9]+$ ]] && channel_ids+=("$_cid")
    done
    IFS=$'\n' read -r -d '' -a channel_ids \
        < <(printf '%s\n' "${channel_ids[@]}" | sort -u; printf '\0') || true

    if [[ ${#channel_ids[@]} -eq 0 ]]; then
        echo "    No channel frame files found, skipping"
        continue
    fi

    for ch in "${channel_ids[@]}"; do
        output="${session_dir}${session_name}-C${ch}-fc${frameCount}.tif"

        if [[ -f "$output" ]]; then
            echo "    C${ch}: already exists, skipping"
            continue
        fi

        frames=( "${session_dir}"*_C${ch}_t*.tif )
        echo "    C${ch}: ${#frames[@]} frames -> $(basename "$output")"

        # ── Pad time-index digits so alphabetic sort = temporal order ─────────
        # Olympus FluoView uses 4-digit indices for frames 0–9999 then switches
        # to 5+ digits at 10000+, making alphabetic sort wrong.  Use perl rename
        # with an /e-evaluated substitution to zero-pad _tNNNN in-place to a
        # uniform width (max(6, digits_needed)) before handing off to the stacker.
        if ! $DRY_RUN && [[ ${#frames[@]} -gt 0 ]]; then
            _sample="${frames[0]}"
            _t="${_sample##*_t}"; _t="${_t%%.*}"
            _maxidx=$(( frameCount - 1 ))
            _needed="${#_maxidx}"
            (( _needed < 6 )) && _needed=6
            if [[ ${#_t} -lt $_needed ]]; then
                echo "    C${ch}: padding time indices → ${_needed} digits..."
                find "$session_dir" -maxdepth 1 -name "*_C${ch}_t*.tif" \
                    | xargs -d '\n' rename \
                        "s/_t(\d+)((?:\.ome)?\.tif)\$/'_t'.sprintf('%0${_needed}d',\$1).\$2/e"
                frames=( "${session_dir}"*_C${ch}_t*.tif )
                if [[ "$ch" == "${channel_ids[0]}" ]]; then
                    _new_master="${master/_t0000.ome.tif/_t$(printf '%0*d' "$_needed" 0).ome.tif}"
                    [[ -f "$_new_master" ]] && master="$_new_master"
                fi
            fi
        fi

        if $DRY_RUN; then
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.tif" \
                --output "$output" \
                --flat-output \
                --keep-sources \
                --preallocate \
                --compression none \
                --dry-run
        else
            python "$STACKER" \
                --input "$session_dir" \
                --pattern "*_C${ch}_t*.tif" \
                --output "$output" \
                --flat-output \
                --keep-sources \
                --preallocate \
                --compression none

            # Verify output before touching sources
            if [[ ! -f "$output" || ! -s "$output" ]]; then
                echo "    C${ch}: ERROR — output file missing or empty, source frames preserved"
                continue
            fi

            # Delete source frames only on explicit --delete-sources
            if $DELETE_SOURCES; then
                echo "    C${ch}: deleting ${#frames[@]} source frame(s)..."
                rm -f "${frames[@]}"
                echo "    C${ch}: deleted"
            fi

            # Update Trial.yaml with OME metadata (only once, on first channel)
            if [[ "$ch" == "${channel_ids[0]}" ]]; then
                yaml_path="${session_dir}${session_name}.yaml"
                if [[ -f "$yaml_path" ]]; then
                    echo "    yaml: updating $yaml_path with OME metadata"
                    python "$OME_META" "$master" --update-yaml "$yaml_path" \
                        || echo "    yaml: WARNING ome_meta update failed"
                else
                    echo "    yaml: $yaml_path not found, skipping OME update"
                fi
            fi
        fi
    done

    echo ""
done
