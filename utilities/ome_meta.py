#!/usr/bin/env python
"""
ome_meta.py — Extract metadata from an OME-TIFF master file
=============================================================
Two modes:

1. Shell eval (default):
   Prints key=value pairs for consumption by shell scripts:
       frameCount=010000
       samplingRate=29p93

2. YAML update (--update-yaml <path>):
   Reads an existing Trial.yaml, fills in acquisition_system fields
   from OME-XML, and writes it back. Creates the file if missing.

Usage:
    python ome_meta.py <master.ome.tif>
    python ome_meta.py <master.ome.tif> --update-yaml /path/to/Trial.yaml

Exit codes:
    0  success
    1  file not found or not a TIFF
    2  OME-XML not present or required fields missing
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

try:
    import tifffile
except ImportError:
    print("Error: tifffile not installed.  Run:  pip install tifffile", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: pyyaml not installed.  Run:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)

OME_NS      = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
OME_NS_2015 = "http://www.openmicroscopy.org/Schemas/OME/2015-01"
OME_NS_2008 = "http://www.openmicroscopy.org/Schemas/OME/2008-02"
ALL_NS = [OME_NS, OME_NS_2015, OME_NS_2008]

# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------
UNIT_TO_S = {
    "s": 1.0, "sec": 1.0, "second": 1.0,
    "ms": 1e-3, "millisecond": 1e-3,
    "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6,
}


def _resolve_ns(root: ET.Element) -> str:
    for candidate in ALL_NS:
        if root.tag.startswith(f"{{{candidate}}}"):
            return candidate
    if root.tag.startswith("{"):
        return root.tag[1:root.tag.index("}")]
    raise ValueError("Unrecognised OME-XML namespace")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_pixels(path: Path) -> dict:
    """Parse OME-XML and return a dict of all useful Pixels fields."""
    with tifffile.TiffFile(str(path)) as tf:
        if not tf.is_ome:
            print(f"Error: {path} does not contain OME-XML metadata", file=sys.stderr)
            sys.exit(2)
        xml_str = tf.ome_metadata

    root = ET.fromstring(xml_str)
    try:
        ns = _resolve_ns(root)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    pixels = root.find(f".//{{{ns}}}Pixels")
    if pixels is None:
        print("Error: <Pixels> element not found in OME-XML", file=sys.stderr)
        sys.exit(2)

    def _int(key):
        v = pixels.get(key)
        return int(v) if v is not None else None

    def _float(key):
        v = pixels.get(key)
        return float(v) if v is not None else None

    # TimeIncrement → sample rate
    time_inc = _float("TimeIncrement") or _float("PhysicalSizeT")
    time_unit = (pixels.get("TimeIncrementUnit") or
                 pixels.get("PhysicalSizeTUnit") or "s").lower()
    factor = UNIT_TO_S.get(time_unit, 1.0)

    sample_rate_hz: Optional[float] = None
    if time_inc and time_inc > 0:
        sample_rate_hz = 1.0 / (time_inc * factor)

    return {
        "size_x":         _int("SizeX"),
        "size_y":         _int("SizeY"),
        "size_z":         _int("SizeZ"),
        "size_t":         _int("SizeT"),
        "size_c":         _int("SizeC"),
        "pixel_type":     pixels.get("PixelType"),
        "physical_size_x": _float("PhysicalSizeX"),
        "physical_size_y": _float("PhysicalSizeY"),
        "time_increment":  time_inc,
        "sample_rate_hz":  sample_rate_hz,
    }


def format_rate_str(hz: float) -> str:
    """30.0 → '30p00',  29.93 → '29p93'"""
    rate_int  = int(hz)
    rate_frac = round((hz - rate_int) * 100)
    return f"{rate_int:02d}p{rate_frac:02d}"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------
def _deep_set(d: dict, keys: list, value):
    """Set d[k1][k2]...[kN] = value, creating intermediate dicts as needed."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def update_yaml(yaml_path: Path, px: dict):
    """Read (or create) Trial.yaml and fill in acquisition_system from OME data."""
    if yaml_path.exists():
        doc = yaml.safe_load(yaml_path.read_text()) or {}
    else:
        doc = {}

    s = doc.setdefault("acquisition_system", {}).setdefault("settings", {})

    # sample_rate
    if px["sample_rate_hz"] is not None:
        s["sample_rate"] = {
            "value": round(px["sample_rate_hz"], 4),
            "units": "Hz",
        }

    # frame dimensions
    s["frame_size"] = {
        "x": px["size_x"],
        "y": px["size_y"],
    }

    # pixel size
    s["pixel_size"] = {
        "x": px["physical_size_x"],
        "y": px["physical_size_y"],
        "units": "um",
    }

    # channels / planes
    s["n_channels"] = px["size_c"]
    s["n_planes"]   = px["size_z"]
    s["n_frames"]   = px["size_t"]
    s["pixel_type"] = px["pixel_type"]

    yaml_path.write_text(yaml.dump(doc, default_flow_style=False, sort_keys=False))
    print(f"Updated: {yaml_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract OME-TIFF metadata for shell use or Trial.yaml update."
    )
    parser.add_argument("tiff", help="Path to master OME-TIFF file")
    parser.add_argument("--update-yaml", metavar="PATH", default=None,
                        help="Trial.yaml to update with OME fields (creates if missing)")
    args = parser.parse_args()

    path = Path(args.tiff)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    px = extract_pixels(path)

    if args.update_yaml:
        update_yaml(Path(args.update_yaml), px)
    else:
        # Shell eval mode — backward compatible
        print(f"frameCount={px['size_t']:06d}")
        if px["sample_rate_hz"] is not None:
            print(f"samplingRate={format_rate_str(px['sample_rate_hz'])}")


if __name__ == "__main__":
    main()
