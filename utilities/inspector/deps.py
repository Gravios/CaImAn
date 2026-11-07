"""Lazy loaders for caiman.utils.event_detection / roi_filter /
event_kinetics, with a fallback to this checkout's files when the caiman
package can't import."""

import os
import sys


def _ed():
    """``caiman.utils.event_detection``, imported lazily.

    Falls back to loading the module file from this checkout when the
    ``caiman`` package itself can't be imported (e.g. demo mode on a machine
    without the compiled CaImAn stack).
    """
    mod = sys.modules.get("_inspector_event_detection")
    if mod is not None:
        return mod
    try:
        from caiman.utils import event_detection as mod
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, "caiman", "utils", "event_detection.py")
        name = "_inspector_event_detection"
        spec = importlib.util.spec_from_file_location(name, p)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # @dataclass resolves its module here
        try:
            spec.loader.exec_module(mod)
        except Exception:
            del sys.modules[name]
            raise
    sys.modules["_inspector_event_detection"] = mod
    return mod


def _rf():
    """``caiman.utils.roi_filter``, imported lazily (same fallback as _ed)."""
    mod = sys.modules.get("_inspector_roi_filter")
    if mod is not None:
        return mod
    try:
        from caiman.utils import roi_filter as mod
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, "caiman", "utils", "roi_filter.py")
        spec = importlib.util.spec_from_file_location("_inspector_roi_filter", p)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    sys.modules["_inspector_roi_filter"] = mod
    return mod


def _ek():
    """``caiman.utils.event_kinetics``, imported lazily (same fallback)."""
    mod = sys.modules.get("_inspector_event_kinetics")
    if mod is not None:
        return mod
    try:
        from caiman.utils import event_kinetics as mod
    except Exception:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, os.pardir, "caiman", "utils", "event_kinetics.py")
        spec = importlib.util.spec_from_file_location("_inspector_event_kinetics", p)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    sys.modules["_inspector_event_kinetics"] = mod
    return mod


__all__ = ['_ed', '_rf', '_ek']
