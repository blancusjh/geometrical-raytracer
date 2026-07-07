"""Shared helper: announce an opened vispy window and bring it to the front.

Interactive viewers block in the Qt event loop, which reads as a silent hang if
the window opened *behind* the terminal (GNOME focus-stealing prevention) or on
an unexpected backend. This prints where the window is and raises it; if no
visible window can be confirmed, it prints an actionable hint instead of
staying silent.
"""

from __future__ import annotations

import sys


def announce_window(canvas, tag: str = "[raytracer]") -> None:
    try:
        from vispy import app as vispy_app

        for _ in range(5):  # let the backend actually map the window
            vispy_app.process_events()
    except Exception:
        pass

    platform = "?"
    visible = None
    try:
        native = canvas.native  # Qt widget
        native.raise_()
        native.activateWindow()
        qt_app = native.window().windowHandle()
        from vispy.app import use_app

        platform = use_app().native.platformName()
        visible = bool(native.isVisible())
        del qt_app
    except Exception:
        pass

    if visible:
        print(f"{tag} Ventana abierta (Qt/{platform}). "
              "Ciérrala o pulsa Q/Escape para salir.", file=sys.stderr, flush=True)
    else:
        print(f"{tag} La ventana no se pudo confirmar como visible "
              f"(backend {platform}). Prueba: QT_QPA_PLATFORM=xcb python ... "
              "y revisa que corres en una sesión gráfica (DISPLAY).",
              file=sys.stderr, flush=True)


__all__ = ["announce_window"]
