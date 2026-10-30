"""Announce an opened vispy window and bring it to the front.

An interactive viewer blocks in the GUI event loop, which reads as a silent hang when the window opened behind
the terminal (focus-stealing prevention) or on a backend that has no window at all; this says which.
"""

from __future__ import annotations

import sys

# vispy falls back to these when no windowed toolkit (PySide6, PyQt, glfw, …) is importable: no window can appear.
_OFFSCREEN_BACKENDS = {"egl", "osmesa"}


def announce_window(canvas, tag: str = "[raytracer]") -> None:
    from vispy import app as vispy_app

    try:
        for _ in range(5):  # let the backend map the window
            vispy_app.process_events()
    except Exception:
        pass
    backend, platform, visible = "?", "?", None
    try:
        backend = vispy_app.use_app().backend_name
    except Exception:
        pass
    if backend.lower() in _OFFSCREEN_BACKENDS:
        print(f"{tag} vispy picked the offscreen '{backend}' backend — no window can ever appear, regardless of "
              "DISPLAY, because no windowed GUI toolkit is installed for it. Install one, e.g.: pip install pyside6",
              file=sys.stderr, flush=True)
        return
    try:
        native = canvas.native  # the Qt widget
        native.raise_()
        native.activateWindow()
        platform = vispy_app.use_app().native.platformName()
        visible = bool(native.isVisible())
    except Exception:
        pass
    if visible:
        print(f"{tag} Window opened ({backend}/{platform}). Close it or press Q/Escape to quit.", file=sys.stderr, flush=True)
    else:
        print(f"{tag} Could not confirm the window is visible (backend {backend}/{platform}). Try: QT_QPA_PLATFORM=xcb "
              "python ... and check that you're running in a graphical session (DISPLAY).", file=sys.stderr, flush=True)


__all__ = ["announce_window"]
