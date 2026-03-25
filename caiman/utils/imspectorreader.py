import numpy as np
import mmap
import os
import logging
import tifffile

logger = logging.getLogger(__name__)


class IMSpectorReader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.offset = None
        self.size_x = None
        self.size_y = None
        self.size_z = None
        self.size_t = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.pixel_size_z = None
        self.slices_count = None
        self.metadata = None

        self._parse_file()

    def _parse_file(self):
        with open(self.file_name, 'rb') as f:
            s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            old_magic_string_idx = s.find(b"CDataStack", 0, 262144)
            if old_magic_string_idx == -1:
                raise ValueError("Not a valid MSR file")

            f.seek(old_magic_string_idx + 14)

            def read_string(f):
                length = int.from_bytes(f.read(1), byteorder='little')
                return f.read(length).decode('cp1252')

            filename = read_string(f)
            date = read_string(f)
            recordingDevice = read_string(f)
            f.read(2)
            imageName = read_string(f)
            if int.from_bytes(f.read(1), byteorder='little') != 0xFF:
                raise ValueError("Failed to find end of top header.")

            length = int.from_bytes(f.read(2), byteorder='little')
            metadata = f.read(length).decode('cp1252')
            metadata_pairs = metadata.split("::")
            if len(metadata_pairs) % 2 != 0:
                metadata_pairs = metadata_pairs[:-1]
            self.metadata = {metadata_pairs[i]: metadata_pairs[i + 1] for i in range(0, len(metadata_pairs), 2)}

            f.read(82)
            temp = f.read(4)
            if temp != b'\x00@\xaf@':
                raise ValueError("@_@ not in place!")

            f.read(112)
            while True:
                length = int.from_bytes(f.read(1), byteorder='little')
                if length == 0:
                    break
                f.read(length)

            f.read(95)
            dc = int.from_bytes(f.read(2), byteorder='little') + 1
            f.read(2)
            temp = int.from_bytes(f.read(4), byteorder='little')
            if temp != 0xFFFFFFFF:
                raise ValueError("Stack is not 16 bit.")

            f.read(16)
            self.pixel_size_x = read_string(f)
            self.pixel_size_y = read_string(f)
            self.pixel_size_z = read_string(f)
            axesUnit2 = None
            length = int.from_bytes(f.read(1), byteorder='little')
            if length != 0:
                axesUnit2 = f.read(length).decode('cp1252')

            f.read(44)
            longLength = int.from_bytes(f.read(4), byteorder='little')
            f.read(longLength * 4)
            f.read(20)
            offset0 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            offset1 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            offset2 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            offset3 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            magicDeviceName = read_string(f)
            f.read(6)
            self.size_x = int.from_bytes(f.read(4), byteorder='little')
            self.size_y = int.from_bytes(f.read(4), byteorder='little')
            self.size_z = int.from_bytes(f.read(4), byteorder='little')
            self.size_t = int.from_bytes(f.read(4), byteorder='little')
            length0 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            length1 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            length2 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            length3 = np.frombuffer(f.read(4), dtype=np.float32)[0]
            self.pixel_size_x = length0
            self.pixel_size_y = length1
            self.pixel_size_z = length2
            axesLabelX = read_string(f)
            axesLabelY = read_string(f)
            axesLabel1 = read_string(f)
            axesLabel2 = read_string(f)
            self.offset = f.tell()
            f.seek(self.offset)
            self.slices_count = self.size_z

    # ------------------------------------------------------------------
    # Slice readers
    # ------------------------------------------------------------------

    def read_slice(self, slice_no):
        if not (0 <= slice_no < self.slices_count):
            raise ValueError("Slice number out of range.")
        with open(self.file_name, 'rb') as f:
            slice_offset = self.offset + ((self.size_x * self.size_y * 2) * slice_no)
            f.seek(slice_offset)
            slice_data = f.read(self.size_x * self.size_y * 2)
            return np.frombuffer(slice_data, dtype=np.uint16).reshape((self.size_y, self.size_x))

    def read_slices(self, start, end):
        if not (0 <= start < self.slices_count) or not (0 <= end < self.slices_count):
            raise ValueError("Slice number out of range.")
        if start > end:
            raise ValueError("Start slice number must be less than or equal to end slice number.")
        slices = []
        for slice_no in range(start, end + 1):
            slices.append(self.read_slice(slice_no))
        return np.array(slices)

    def read_range(self, indices):
        slices = []
        for slice_no in indices:
            slices.append(self.read_slice(slice_no))
        return np.array(slices)

    def read_whole(self):
        return self.read_slices(0, self.slices_count - 1)

    # ------------------------------------------------------------------
    # BigTIFF export
    # ------------------------------------------------------------------

    def to_bigtiff(
        self,
        out_path: str | None = None,
        *,
        skip_if_exists: bool = True,
        log_every: int = 500,
    ) -> str:
        """Write all slices to a BigTIFF file, one frame at a time.

        Frames are written sequentially without loading the whole stack into
        RAM, keeping peak memory at a single frame (~size_x * size_y * 2 bytes).

        Pixel sizes are embedded as ImageJ resolution metadata (µm assumed;
        adjust ``unit`` below if your acquisition uses different units).

        Args:
            out_path:       Destination path.  Defaults to the source path
                            with the extension replaced by ``.tif``.
            skip_if_exists: If True and ``out_path`` already exists, return
                            immediately without overwriting.
            log_every:      Emit a progress log line every this many frames.

        Returns:
            Absolute path of the written (or skipped) file.
        """
        if out_path is None:
            out_path = os.path.splitext(self.file_name)[0] + ".tif"

        out_path = os.path.abspath(out_path)

        if skip_if_exists and os.path.exists(out_path):
            logger.info("to_bigtiff: output already exists, skipping — %s", out_path)
            return out_path

        n = self.slices_count
        logger.info(
            "to_bigtiff: converting %s  (%d × %d, %d frames) → %s",
            self.file_name, self.size_x, self.size_y, n, out_path,
        )

        # pixel_size_x / _y are stored in metres by IMSpector; convert to µm
        # for the ImageJ resolution tag (1 / µm_per_pixel).
        try:
            px_um = float(self.pixel_size_x) * 1e6   # m → µm
            py_um = float(self.pixel_size_y) * 1e6
            resolution = (1.0 / px_um, 1.0 / py_um)
            resolution_unit = "MICROMETER"
        except (TypeError, ValueError, ZeroDivisionError):
            logger.warning("to_bigtiff: could not convert pixel sizes; omitting resolution tag")
            resolution = None
            resolution_unit = None

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with tifffile.TiffWriter(out_path, bigtiff=True) as tif:
            for i in range(n):
                frame = self.read_slice(i)   # (size_y, size_x) uint16, ~one frame in RAM

                write_kwargs = dict(
                    photometric="minisblack",
                    contiguous=True,        # pack IFDs tightly → fast sequential reads
                )

                if i == 0 and resolution is not None:
                    write_kwargs["resolution"] = resolution
                    if resolution_unit is not None:
                        write_kwargs["resolutionunit"] = resolution_unit

                tif.write(frame, **write_kwargs)

                if (i + 1) % log_every == 0 or (i + 1) == n:
                    logger.info("  frame %d / %d", i + 1, n)

        logger.info("to_bigtiff: done — %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def local_correlations(self, swap_dim: bool = False, order_mean=1) -> np.ndarray:
        """Computes the correlation image for the input dataset Y

        Args:
            Y:  np.ndarray (3D or 4D)
                Input movie data in 3D or 4D format

            eight_neighbours: Boolean
                Use 8 neighbors if true, and 4 if false for 3D data (default = True)
                Use 6 neighbors for 4D data, irrespectively

            swap_dim: Boolean
                True indicates that time is listed in the last axis of Y (matlab format)
                and moves it in the front

            order_mean: (undocumented)

        Returns:
            rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels
        """
        Y = self.read_slices(0, 500)
        if swap_dim:
            Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

        rho = np.zeros(np.shape(Y)[1:])
        w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

        rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
        rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

        if order_mean == 0:
            rho = np.ones(np.shape(Y)[1:])
            rho_h = rho_h
            rho_w = rho_w
            rho[:-1, :] = rho[:-1, :] * rho_h
            rho[1:,  :] = rho[1:,  :] * rho_h
            rho[:, :-1] = rho[:, :-1] * rho_w
            rho[:,  1:] = rho[:,  1:] * rho_w
        else:
            rho[:-1, :] = rho[:-1, :] + rho_h**(order_mean)
            rho[1:,  :] = rho[1:,  :] + rho_h**(order_mean)
            rho[:, :-1] = rho[:, :-1] + rho_w**(order_mean)
            rho[:,  1:] = rho[:,  1:] + rho_w**(order_mean)

        if Y.ndim == 4:
            rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
            rho[:, :, :-1] = rho[:, :, :-1] + rho_d
            rho[:, :, 1:] = rho[:, :, 1:] + rho_d

            neighbors = 6 * np.ones(np.shape(Y)[1:])
            neighbors[0]        = neighbors[0]        - 1
            neighbors[-1]       = neighbors[-1]       - 1
            neighbors[:,     0] = neighbors[:,     0] - 1
            neighbors[:,    -1] = neighbors[:,    -1] - 1
            neighbors[:,  :, 0] = neighbors[:,  :, 0] - 1
            neighbors[:, :, -1] = neighbors[:, :, -1] - 1

        else:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:,]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:,]), axis=0)

            if order_mean == 0:
                rho_d1 = rho_d1
                rho_d2 = rho_d2
                rho[:-1, :-1] = rho[:-1, :-1] * rho_d2
                rho[1:,   1:] = rho[1:,   1:] * rho_d1
                rho[1:,  :-1] = rho[1:,  :-1] * rho_d1
                rho[:-1,  1:] = rho[:-1,  1:] * rho_d2
            else:
                rho[:-1, :-1] = rho[:-1, :-1] + rho_d2**(order_mean)
                rho[1:,   1:] = rho[1:,   1:] + rho_d1**(order_mean)
                rho[1:,  :-1] = rho[1:,  :-1] + rho_d1**(order_mean)
                rho[:-1,  1:] = rho[:-1,  1:] + rho_d2**(order_mean)

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0,   :] = neighbors[0,   :] - 3
            neighbors[-1,  :] = neighbors[-1,  :] - 3
            neighbors[:,   0] = neighbors[:,   0] - 3
            neighbors[:,  -1] = neighbors[:,  -1] - 3
            neighbors[0,   0] = neighbors[0,   0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1,  0] = neighbors[-1,  0] + 1
            neighbors[0,  -1] = neighbors[0,  -1] + 1

        if order_mean == 0:
            rho = np.power(rho, 1. / neighbors)
        else:
            rho = np.power(np.divide(rho, neighbors), 1 / order_mean)

        return rho


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def convert_msr_to_bigtiff(
    msr_path: str,
    out_path: str | None = None,
    *,
    skip_if_exists: bool = True,
    log_every: int = 500,
) -> str:
    """Parse *msr_path* and write it as a BigTIFF stack.

    Thin wrapper around ``IMSpectorReader.to_bigtiff``; useful for batch
    conversion scripts::

        from imspectorreader import convert_msr_to_bigtiff
        convert_msr_to_bigtiff("/data/session/recording.msr")

    Returns the path of the output file.
    """
    reader = IMSpectorReader(msr_path)
    return reader.to_bigtiff(out_path, skip_if_exists=skip_if_exists, log_every=log_every)


# ---------------------------------------------------------------------------
# CLI entry point:  python imspectorreader.py recording.msr [output.tif]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Convert an IMSpector .msr file to a BigTIFF stack."
    )
    parser.add_argument("msr", help="Input .msr file")
    parser.add_argument(
        "tif",
        nargs="?",
        default=None,
        help="Output .tif path (default: same stem as input, same directory)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        metavar="N",
        help="Log a progress line every N frames (default: 500)",
    )
    args = parser.parse_args()

    out = convert_msr_to_bigtiff(
        args.msr,
        args.tif,
        skip_if_exists=not args.overwrite,
        log_every=args.log_every,
    )
    print(out)
