#!/usr/bin/env python

"""
Functions related to the creation and management of the "cluster",
meaning the framework for distributed computation.

We put arrays on disk as raw bytes, extending along the first dimension.
Alongside each array x we ensure the value x.dtype which stores the data type.
"""

import ipyparallel
import logging
import multiprocessing
import numpy as np
import os
import platform
import psutil
import shlex
import subprocess
import sys
import time
from typing import Any, Optional, Union

def extract_patch_coordinates(dims: tuple,
                              rf: Union[list, tuple],
                              stride: Union[list[int], tuple],
                              border_pix: int = 0,
                              indices=[slice(None)] * 2) -> tuple[list, list]:
    """
    Partition the FOV in patches
    and return the indexed in 2D and 1D (flatten, order='F') formats

    Args:
        dims: tuple of int
            dimensions of the original matrix that will be divided in patches

        rf: tuple of int
            radius of receptive field, corresponds to half the size of the square patch

        stride: tuple of int
            degree of overlap of the patches
    """

    # TODO: Find a new home for this function
    sl_start = [0 if sl.start is None else sl.start for sl in indices]
    sl_stop = [dim if sl.stop is None else sl.stop for (sl, dim) in zip(indices, dims)]
    sl_step = [1 for sl in indices]    # not used
    dims_large = dims
    dims = np.minimum(np.array(dims) - border_pix, sl_stop) - np.maximum(border_pix, sl_start)

    coords_flat = []
    shapes = []
    iters = [list(range(rf[i], dims[i] - rf[i], 2 * rf[i] - stride[i])) + [dims[i] - rf[i]] for i in range(len(dims))]

    coords = np.empty(list(map(len, iters)) + [len(dims)], dtype=object)
    for count_0, xx in enumerate(iters[0]):
        coords_x = np.arange(xx - rf[0], xx + rf[0] + 1)
        coords_x = coords_x[(coords_x >= 0) & (coords_x < dims[0])]
        coords_x += border_pix * 0 + np.maximum(sl_start[0], border_pix)

        for count_1, yy in enumerate(iters[1]):
            coords_y = np.arange(yy - rf[1], yy + rf[1] + 1)
            coords_y = coords_y[(coords_y >= 0) & (coords_y < dims[1])]
            coords_y += border_pix * 0 + np.maximum(sl_start[1], border_pix)

            if len(dims) == 2:
                idxs = np.meshgrid(coords_x, coords_y)

                coords[count_0, count_1] = idxs
                shapes.append(idxs[0].shape[::-1])

                coords_ = np.ravel_multi_index(idxs, dims_large, order='F')
                coords_flat.append(coords_.flatten())
            else:      # 3D data

                if border_pix > 0:
                    raise Exception(
                        'The parameter border pix must be set to 0 for 3D data since border removal is not implemented')

                for count_2, zz in enumerate(iters[2]):
                    coords_z = np.arange(zz - rf[2], zz + rf[2] + 1)
                    coords_z = coords_z[(coords_z >= 0) & (coords_z < dims[2])]
                    idxs = np.meshgrid(coords_x, coords_y, coords_z)
                    shps = idxs[0].shape
                    shapes.append([shps[1], shps[0], shps[2]])
                    coords[count_0, count_1, count_2] = idxs
                    coords_ = np.ravel_multi_index(idxs, dims, order='F')
                    coords_flat.append(coords_.flatten())

    for i, c in enumerate(coords_flat):
        assert len(c) == np.prod(shapes[i])

    return list(map(np.sort, coords_flat)), shapes

def start_server(ipcluster: str = "ipcluster", ncpus: int = None) -> None:
    """
    programmatically start the ipyparallel server

    Args:
        ncpus
            number of processors

        ipcluster
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe"
            Default: "ipcluster"
    """
    logger = logging.getLogger("caiman")
    logger.info("Starting cluster...")
    if ncpus is None:
        ncpus = psutil.cpu_count()

    if ipcluster == "ipcluster":
        subprocess.Popen(f"ipcluster start -n {ncpus}", shell=True, close_fds=(os.name != 'nt'))
    else:
        subprocess.Popen(shlex.split(f"{ipcluster} start -n {ncpus}"),
                         shell=True,
                         close_fds=(os.name != 'nt'))
    time.sleep(1.5)
    # Check that all processes have started
    client = ipyparallel.Client()
    time.sleep(1.5)
    while len(client) < ncpus:
        sys.stdout.write(".")                              # Give some visual feedback of things starting
        sys.stdout.flush()                                 # (de-buffered)
        time.sleep(0.5)
    logger.debug('Making sure everything is up and running')
    client.direct_view().execute('__a=1', block=True)      # when done on all, we're set to go

def stop_server(ipcluster: str = 'ipcluster', pdir: str = None, profile: str = None, dview=None) -> None:
    """
    programmatically stops the ipyparallel server

    Args:
        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows
            Default: "ipcluster"a

        pdir : Undocumented
        profile: Undocumented
        dview: Undocumented

    """
    logger = logging.getLogger("caiman")
    if 'multiprocessing' in str(type(dview)):
        dview.terminate()
    else:
        logger.info("Stopping cluster...")

        if ipcluster == "ipcluster":
            proc = subprocess.Popen("ipcluster stop",
                                    shell=True,
                                    stderr=subprocess.PIPE,
                                    close_fds=(os.name != 'nt'))
        else:
            proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                    shell=True,
                                    stderr=subprocess.PIPE,
                                    close_fds=(os.name != 'nt'))

        line_out = proc.stderr.readline()
        if b'CRITICAL' in line_out:
            logger.info("No cluster to stop...")
        elif b'Stopping' in line_out:
            st = time.time()
            logger.debug('Waiting for cluster to stop...')
            while (time.time() - st) < 4:
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
        else:
            logger.error(line_out)
            logger.error('**** Unrecognized syntax in ipcluster output, waiting for server to stop anyways ****')

        proc.stderr.close()

    logger.info("stop_cluster(): done")

# ---------------------------------------------------------------------------
# Spawn-safe logging for pool workers.
#
# Spawned workers start with a clean interpreter — no handlers on the caiman
# logger.  Rather than opening the shared log file directly (which causes
# interleaved writes across concurrent workers), each worker accumulates
# LogRecords in a _WorkerBufferingHandler and flushes atomically at the end
# of each patch using fcntl.flock for mutual exclusion.
# ---------------------------------------------------------------------------

class _WorkerBufferingHandler(logging.Handler):
    """Logging handler that stores records in memory for deferred flushing.

    Attached to the caiman logger inside each spawned worker.  Records
    accumulate until flush_worker_log() is called at patch completion,
    at which point they are written to the shared log file atomically
    under an exclusive flock.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def flush_to_file(self, log_file: str, formatter: logging.Formatter) -> None:
        """Write buffered records to *log_file* under an exclusive flock.

        Blocks until the lock is available, writes all records as a single
        contiguous block, then releases the lock.  Clears the buffer
        regardless of whether the write succeeded.
        """
        import fcntl
        records, self.records = self.records, []
        if not records or not log_file:
            return
        lines = "".join(formatter.format(r) + "\n" for r in records)
        try:
            with open(log_file, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    fh.write(lines)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            pass  # best-effort; never raise inside a pool worker


def _worker_logging_init(log_params: dict) -> None:
    """Pool initializer: set BLAS threads and configure logging.

    Receives log_params via initargs — the module-level approach does not
    work with spawn because the child re-imports the module fresh and never
    sees values set on the parent's copy.

    Sets 2 BLAS threads per worker so 8 workers × 2 threads fills all 16
    cores.  Must be set before any BLAS import in the worker process.
    Records accumulate in memory and are flushed atomically at patch
    completion by flush_worker_log().
    """
    # Set BLAS thread count using threadpoolctl — works at runtime even
    # after numpy/scipy have already imported and initialized their thread
    # pools. Setting env vars here is too late (BLAS already locked pool
    # during module import, before this initializer runs).
    _n_blas = log_params.get('blas_threads', 1) if log_params else 1
    if _n_blas > 1:
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=_n_blas)
        except ImportError:
            # fallback: set env vars for any BLAS that re-checks them
            import os as _os_w
            for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS'):
                _os_w.environ[_var] = str(_n_blas)
    # Stash in module scope so flush_worker_log() can find log_file later.
    global _worker_log_params
    _worker_log_params = log_params
    p = log_params
    if not p:
        return
    log = logging.getLogger("caiman")
    if any(isinstance(h, _WorkerBufferingHandler) for h in log.handlers):
        return  # already configured (maxtasksperchild=1 never reuses, but be safe)
    log.setLevel(p.get("level", logging.INFO))
    buf = _WorkerBufferingHandler()
    buf.setFormatter(logging.Formatter(
        p.get("fmt",     "%(asctime)s %(relativeCreated)12d "
                         "[%(filename)s:%(funcName)20s():%(lineno)s] "
                         "[%(process)d] %(message)s"),
        datefmt=p.get("datefmt", "%Y-%m-%d %H:%M:%S"),
    ))
    log.addHandler(buf)
    # Stderr handler: WARNING+ only, so critical errors surface immediately.
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s/%(process)d] %(message)s",
        datefmt="%H:%M:%S"))
    sh.setLevel(logging.WARNING)
    log.addHandler(sh)


_worker_log_params: dict = {}   # stashed by _worker_logging_init in each worker


def flush_worker_log() -> None:
    """Flush buffered log records to the shared log file.

    Call once at the end of each patch job (both success and exception
    paths).  Locates the _WorkerBufferingHandler on the caiman logger,
    acquires an exclusive flock on the log file, writes all pending
    records as a single block, then releases the lock.
    """
    log = logging.getLogger("caiman")
    log_file = _worker_log_params.get("log_file")
    for h in log.handlers:
        if isinstance(h, _WorkerBufferingHandler):
            h.flush_to_file(log_file, h.formatter)
            return


def _collect_log_params() -> dict:
    """Snapshot the parent caiman logger config for passing to workers."""
    log = logging.getLogger("caiman")
    log_file = None
    fmt      = ("%(asctime)s %(relativeCreated)12d "
                "[%(filename)s:%(funcName)20s():%(lineno)s] "
                "[%(process)d] %(message)s")
    datefmt  = "%Y-%m-%d %H:%M:%S"
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            log_file = h.baseFilename
            if h.formatter:
                fmt     = h.formatter._fmt    or fmt
                datefmt = h.formatter.datefmt or datefmt
            break
    return {"level": log.level, "log_file": log_file, "fmt": fmt, "datefmt": datefmt}


def _worker_cuda_reset() -> None:
    """Pool worker initializer: reset CUDA context inherited from parent fork.

    cp.cuda.Device().reset() calls cudaDeviceReset() which requires the
    runtime to be functional — it silently fails when the inherited context
    is already in cudaErrorInitializationError state.

    Instead we call the CUDA driver API (cuInit + cuDevicePrimaryCtxReset)
    directly via ctypes, bypassing the broken runtime state entirely.
    Falls back gracefully if libcuda is unavailable or CuPy not installed.
    """
    try:
        import ctypes, ctypes.util
        _libcuda = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        _libcuda.cuInit(0)
        _libcuda.cuDevicePrimaryCtxReset(0)
    except Exception:
        pass
    try:
        import cupy as cp
        cp.cuda.Device(0).use()  # force CuPy to re-initialise on device 0
    except Exception:
        pass


def setup_cluster(backend:str = 'multiprocessing',
                  n_processes:Optional[int] = None,
                  single_thread:bool = False,
                  ignore_preexisting:bool = False,
                  maxtasksperchild:int = None) -> tuple[Any, Any, Optional[int]]:
    """
    Setup and/or restart a parallel cluster.

    Args:
        backend:
            One of:
                'multiprocessing' - Use multiprocessing library
                'ipyparallel' - Use ipyparallel instead (better on Windows?)
                'single' - Don't be parallel (good for debugging, slow)

            Most backends will try, by default, to stop a running cluster if
            it is running before setting up a new one, or throw an error if
            they find one.
        n_processes:
            Sets number of processes to use. If None, is set automatically. 
        single_thread:
            Deprecated alias for the 'single' backend.
        ignore_preexisting:
            If True, ignores the existence of an already running multiprocessing
            pool (which usually indicates a previously-started CaImAn cluster)
        maxtasksperchild:
            Only used for multiprocessing, default None (number of tasks a worker process can 
            complete before it will exit and be replaced with a fresh worker process).
            
    Returns:
        c:
            ipyparallel.Client object; only used for ipyparallel backends, else None
        dview:
            multicore processing engine that is used for parallel processing. 
            If backend is 'multiprocessing' then dview is Pool object.
            If backend is 'ipyparallel' then dview is a DirectView object. 
        n_processes:
            number of workers in dview. None means single core mode in use. 
    """

    logger = logging.getLogger("caiman")
    sys.stdout.flush() # XXX Unsure why we do this
    if n_processes is None:
        n_processes = np.maximum(int(psutil.cpu_count() - 1), 1)

    if backend == 'multiprocessing' or backend == 'local':
        if backend == 'local':
            logger.warning('The local backend is an alias for the multiprocessing backend, and the alias may be removed in some future version of Caiman')
        if len(multiprocessing.active_children()) > 0:
            if ignore_preexisting:
                logger.warning('Found an existing multiprocessing pool. '
                               'This is often indicative of an already-running CaImAn cluster. '
                               'You have configured the cluster setup to not raise an exception.')
            else:
                raise Exception(
                    'A cluster is already running. Terminate with dview.terminate() if you want to restart.')
        # Use spawn so workers start with a clean process — no inherited
        # CUDA state, no vecLib thread-count issues on macOS (supersedes
        # the old forkserver workaround for Jupyter/Spyder on Darwin).
        # Pass log params via initargs — module-level vars are not
        # inherited by spawned workers.
        c = None
        _lp = _collect_log_params()
        _spawn_ctx = multiprocessing.get_context('spawn')
        dview = _spawn_ctx.Pool(
            n_processes,
            maxtasksperchild = maxtasksperchild,
            initializer      = _worker_logging_init,
            initargs         = (_lp,),
        )

    elif backend == 'ipyparallel':
        stop_server()
        start_server(ncpus=n_processes)
        c = ipyparallel.Client()
        logger.info(f'Started ipyparallel cluster: Using {len(c)} processes')
        dview = c[:len(c)]

    elif backend == "single" or single_thread:
        if single_thread:
            logger.warning('The single_thread flag to setup_cluster() is deprecated and may be removed in the future')
        dview = None
        c = None
        n_processes = 1

    else:
        raise Exception('Unknown Backend')

    return c, dview, n_processes
