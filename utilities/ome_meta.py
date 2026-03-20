#!/usr/bin/env python
"""
ome_meta.py — Extract frameCount and samplingRate from an OME-TIFF master file
===============================================================================
Reads the OME-XML embedded in the first page of a TIFF and prints key=value
pairs to stdout for consumption by shell scripts:

    frameCount=010000
    samplingRate=30p00

Usage:
    python ome_meta.py <path/to/master.ome.tif>

Exit codes:
    0  success
    1  file not found or not a TIFF
    2  OME-XML not present or required fields missing
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import tifffile
except ImportError:
    print("Error: tifffile not installed.  Run:  pip install tifffile", file=sys.stderr)
    sys.exit(1)


OME_NS      = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_NS_2015 = "http://www.openmicroscopy.org/Schemas/OME/2015-01"
OME_NS_2008 = "http://www.openmicroscopy.org/Schemas/OME/2008-02"

ALL_NS = [OME_NS, OME_NS_2015, OME_NS_2008]


def parse_ome_xml(path: Path):
    with tifffile.TiffFile(str(path)) as tf:
        if not tf.is_ome:
            print(f"Error: {path} does not contain OME-XML metadata", file=sys.stderr)
            sys.exit(2)
        xml_str = tf.ome_metadata

    root = ET.fromstring(xml_str)

    # Resolve namespace — try all known OME schema versions
    ns = None
    for candidate in ALL_NS:
        if root.tag.startswith(f"{{{candidate}}}"):
            ns = candidate
            break
    if ns is None:
        # Last resort: extract from root tag directly
        if root.tag.startswith("{"):
            ns = root.tag[1:root.tag.index("}")]
        else:
            print(f"Error: unrecognised OME-XML namespace in {path}", file=sys.stderr)
            sys.exit(2)

    # --- frameCount: SizeT on the first Image/Pixels element ---
    pixels = root.find(f".//{{{ns}}}Pixels")
    if pixels is None:
        print("Error: <Pixels> element not found in OME-XML", file=sys.stderr)
        sys.exit(2)

    size_t = pixels.get("SizeT")
    if size_t is None:
        print("Error: SizeT attribute missing from <Pixels>", file=sys.stderr)
        sys.exit(2)

    frame_count = int(size_t)

    # --- samplingRate: derived from TimeIncrement (seconds per frame) ---
    # OME spec: TimeIncrement in PhysicalSizeTUnit (default seconds)
    time_inc = pixels.get("TimeIncrement")
    time_unit = pixels.get("TimeIncrementUnit", "s").lower()

    if time_inc is None:
        # Fallback: some Leica files store it as PhysicalSizeT
        time_inc = pixels.get("PhysicalSizeT")
        time_unit = pixels.get("PhysicalSizeTUnit", "s").lower()

    if time_inc is None:
        print("Error: TimeIncrement / PhysicalSizeT not found in OME-XML", file=sys.stderr)
        sys.exit(2)

    time_inc_s = float(time_inc)

    # Normalise to seconds
    unit_to_s = {
        "s": 1.0, "sec": 1.0, "second": 1.0,
        "ms": 1e-3, "millisecond": 1e-3,
        "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6,
    }
    factor = unit_to_s.get(time_unit, 1.0)
    time_inc_s *= factor

    if time_inc_s <= 0:
        print(f"Error: non-positive TimeIncrement: {time_inc_s}", file=sys.stderr)
        sys.exit(2)

    sample_rate_hz = 1.0 / time_inc_s

    # Format: integer part + 2 decimal digits, 'p' as decimal separator
    # e.g. 30.0 → "30p00",  29.97 → "29p97"
    rate_int  = int(sample_rate_hz)
    rate_frac = round((sample_rate_hz - rate_int) * 100)
    rate_str  = f"{rate_int:02d}p{rate_frac:02d}"

    # Output for shell eval
    print(f"frameCount={frame_count:06d}")
    print(f"samplingRate={rate_str}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <master.ome.tif>", file=sys.stderr)
        sys.exit(1)

    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    parse_ome_xml(p)
