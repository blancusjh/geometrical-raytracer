"""Shared helper: announce an opened vispy window and bring it to the front.

Interactive viewers block in the Qt event loop, which reads as a silent hang if
the window opened *behind* the terminal (GNOME focus-stealing prevention) or on
an unexpected backend. This prints where the window is and raises it; if no
visible window can be confirmed, it prints an actionable hint instead of
staying silent.
"""

from __future__ import annotations

import sys


# vispy backends with no real window at all: it silently falls back to one
# of these when no windowed GUI toolkit (PySide6/PyQt5/PyQt6/glfw/...) is
# importable, which otherwise looks identical to a window hidden offscreen.
_OFFSCREEN_BACKENDS = {"egl", "osmesa"}


def announce_window(canvas, tag: str = "[raytracer]") -> None:
    try:
        from vispy import app as vispy_app

        for _ in range(5):  # let the backend actually map the window
            vispy_app.process_events()
    except Exception:
        pass

    from vispy.app import use_app

    backend_name = "?"
    try:
        backend_name = use_app().backend_name
    except Exception:
        pass

    if backend_name.lower() in _OFFSCREEN_BACKENDS:
        print(f"{tag} vispy picked the offscreen '{backend_name}' backend — no window "
              "can ever appear, regardless of DISPLAY. This means no windowed GUI "
              "toolkit is installed for it to use instead. Install one, e.g.: "
              "pip install pyside6", file=sys.stderr, flush=True)
        return

    platform = "?"
    visible = None
    try:
        native = canvas.native  # Qt widget
        native.raise_()
        native.activateWindow()
        qt_app = native.window().windowHandle()
        platform = use_app().native.platformName()
        visible = bool(native.isVisible())
        del qt_app
    except Exception:
        pass

    if visible:
        print(f"{tag} Window opened ({backend_name}/{platform}). "
              "Close it or press Q/Escape to quit.", file=sys.stderr, flush=True)
    else:
        print(f"{tag} Could not confirm the window is visible "
              f"(backend {backend_name}/{platform}). Try: QT_QPA_PLATFORM=xcb python ... "
              "and check that you're running in a graphical session (DISPLAY).",
              file=sys.stderr, flush=True)


__all__ = ["announce_window"]
