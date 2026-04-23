#!/usr/bin/env bash
# png2mp4_lossless.sh  —  lossless .etrk.png stack → MP4 / MKV
#
# Filename format expected:
#   <prefix>-<YYYYMMDD>-<HHMMSS>-<mmm>-<NNNNNNN>.etrk.png
#
# Frames are ordered by the 7-digit frame counter (last numeric field).
# Optimised for large stacks (100k+ frames) at high frame rates.
#
# Usage:
#   ./png2mp4_lossless.sh [OPTIONS] <input_dir> <output.mp4>
#
# Options:
#   -r FPS      Frame rate            (default: 100)
#   -c CODEC    x264 | x264rgb | ffv1 (default: x264)
#   -p PRESET   FFmpeg preset for x264/x264rgb: ultrafast ... veryslow
#                 (default: fast — at CRF 0 all presets are bit-exact,
#                  faster preset = faster encode, marginally larger file)
#   -h          Show this help

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
FPS=100
CODEC="x264"
PRESET="fast"

# ── argument parsing ──────────────────────────────────────────────────────────
usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
while getopts "r:c:p:h" opt; do
    case $opt in
        r) FPS="$OPTARG" ;;
        c) CODEC="$OPTARG" ;;
        p) PRESET="$OPTARG" ;;
        h) usage ;;
        *) echo "Unknown option -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

INPUT_DIR="${1:?ERROR: input_dir required. Run with -h for help.}"
OUTPUT="${2:?ERROR: output file required. Run with -h for help.}"

# ── resolve input dir once (avoid per-file realpath) ─────────────────────────
ABS_DIR=$(realpath "$INPUT_DIR")
[[ -d "$ABS_DIR" ]] || { echo "ERROR: '$ABS_DIR' is not a directory."; exit 1; }

# ── build sorted concat list ──────────────────────────────────────────────────
# Sort numerically by the 7-digit frame counter (7th dash-delimited field).
# awk builds the concat entry in one pass; no subprocess per frame.
TMPLIST=$(mktemp /tmp/ffmpeg_concat_XXXXXX.txt)
trap 'rm -f "$TMPLIST"' EXIT

echo "Scanning for .etrk.png frames in: $ABS_DIR"

find "$ABS_DIR" -maxdepth 1 -name "*.etrk.png" -printf "%f\n" \
    | sort -t'-' -k7 -n \
    | awk -v dir="$ABS_DIR" -v fps="$FPS" \
        '{ print "file " dir "/" $0 "\nduration " 1/fps }' \
    > "$TMPLIST"

NFRAMES=$(wc -l < "$TMPLIST")
DURATION=$(awk "BEGIN{printf \"%.1f\", $NFRAMES/$FPS}")
printf "Found %d frames  (%.1f s @ %d Hz)\n" "$NFRAMES" "$DURATION" "$FPS"

[[ $NFRAMES -eq 0 ]] && { echo "ERROR: no .etrk.png files found."; exit 1; }

# ── codec settings ────────────────────────────────────────────────────────────
case "$CODEC" in
    x264)
        # Lossless H.264, yuv420p — plays in VLC, QuickTime, browsers.
        # NOTE: yuv420p does chroma subsampling. If exact RGB matters, use x264rgb.
        # For grayscale tracking frames this is bit-exact (luma is preserved fully).
        VCODEC_ARGS=(-c:v libx264 -crf 0 -preset "$PRESET" -pix_fmt yuv420p)
        ;;
    x264rgb)
        # Lossless H.264 RGB — exact pixel values, ~2x larger than x264
        VCODEC_ARGS=(-c:v libx264rgb -crf 0 -preset "$PRESET" -pix_fmt rgb24)
        ;;
    ffv1)
        # FFV1 level 3 — true lossless archival, fastest encode, requires .mkv
        VCODEC_ARGS=(-c:v ffv1 -level 3 -g 1 -slices 24 -slicecrc 1)
        OUTPUT="${OUTPUT%.*}.mkv"
        echo "FFV1 selected -> output: $OUTPUT"
        ;;
    *)
        echo "ERROR: unknown codec '$CODEC'. Choose x264, x264rgb, or ffv1." >&2
        exit 1 ;;
esac

# ── encode ────────────────────────────────────────────────────────────────────
echo "Encoding: codec=$CODEC  preset=$PRESET  fps=$FPS  -> $OUTPUT"
echo "This may take several minutes for large stacks."
echo ""

ffmpeg -y \
    -f concat -safe 0 \
    -r "$FPS" \
    -i "$TMPLIST" \
    -r "$FPS" \
    "${VCODEC_ARGS[@]}" \
    -movflags +faststart \
    "$OUTPUT"

echo ""
echo "Done: $OUTPUT"
SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "  Size    : $SIZE"
echo "  Frames  : $NFRAMES"
echo "  Duration: ${DURATION} s @ ${FPS} Hz"
