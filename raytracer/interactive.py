"""Interactive controls for ray tracing visualization."""

from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass

import numpy as np
from vispy import scene


@dataclass
class SliderConfig:
    """Configuration for a single slider."""
    name: str
    min_val: float
    max_val: float
    initial: float
    step: float | None = None
    format_str: str = "{:.2f}"


class InteractiveController:
    """Manages interactive sliders and parameter updates."""

    def __init__(self, canvas: scene.SceneCanvas, update_callback: Callable[[dict[str, Any]], None]):
        """
        Initialize interactive controller.

        Parameters
        ----------
        canvas : SceneCanvas
            The VisPy canvas to attach controls to
        update_callback : callable
            Function called with dict of parameter values when any slider changes
        """
        self.canvas = canvas
        self.update_callback = update_callback
        self.sliders: dict[str, Any] = {}
        self.values: dict[str, float] = {}
        self.widgets: list[Any] = []

        # Create a widget container (uses VisPy's built-in text overlay for now)
        # For true sliders, we'll use keyboard controls
        self._setup_keyboard_controls()
        self._active_param: str | None = None
        self._param_list: list[str] = []

    def add_slider(self, config: SliderConfig) -> None:
        """Add a slider control."""
        self.values[config.name] = config.initial
        self.sliders[config.name] = config
        self._param_list.append(config.name)
        if self._active_param is None:
            self._active_param = config.name

    def _setup_keyboard_controls(self) -> None:
        """Setup keyboard controls for slider adjustment."""
        @self.canvas.events.key_press.connect
        def on_key(event):
            if self._active_param is None:
                return

            config = self.sliders.get(self._active_param)
            if config is None:
                return

            current_val = self.values[self._active_param]
            step = config.step or (config.max_val - config.min_val) / 100.0
            changed = False

            # Arrow keys to adjust value
            if event.key == 'Up' or event.key == 'Right':
                new_val = min(current_val + step, config.max_val)
                if new_val != current_val:
                    self.values[self._active_param] = new_val
                    changed = True
            elif event.key == 'Down' or event.key == 'Left':
                new_val = max(current_val - step, config.min_val)
                if new_val != current_val:
                    self.values[self._active_param] = new_val
                    changed = True

            # Tab to cycle through parameters
            elif event.key == 'Tab':
                if self._param_list:
                    idx = self._param_list.index(self._active_param)
                    idx = (idx + 1) % len(self._param_list)
                    self._active_param = self._param_list[idx]
                    print(f"Active parameter: {self._active_param} = {self.values[self._active_param]}")

            # Space to trigger update
            elif event.key == ' ':
                changed = True

            if changed:
                print(f"{self._active_param} = {config.format_str.format(self.values[self._active_param])}")
                self.update_callback(self.values.copy())

    def get_value(self, name: str) -> float:
        """Get current value of a parameter."""
        return self.values.get(name, 0.0)

    def get_all_values(self) -> dict[str, float]:
        """Get all current parameter values."""
        return self.values.copy()

    def print_help(self) -> None:
        """Print help text for controls."""
        print("\n=== Interactive Controls ===")
        print("Tab       : Switch active parameter")
        print("Up/Right  : Increase value")
        print("Down/Left : Decrease value")
        print("Space     : Force update")
        print("\nParameters:")
        for name, config in self.sliders.items():
            marker = "(*)" if name == self._active_param else "   "
            print(f"{marker} {name}: {config.format_str.format(self.values[name])} "
                  f"[{config.min_val}, {config.max_val}]")
        print("===========================\n")


class SliderWidget:
    """
    Visual slider widget overlaid on the canvas.

    This is a simple text-based implementation. For graphical sliders,
    VisPy's Widget system or ImGui integration would be better.
    """

    def __init__(
        self,
        parent: scene.SceneCanvas,
        config: SliderConfig,
        position: tuple[float, float] = (10, 10),
        on_change: Callable[[float], None] | None = None,
    ):
        self.config = config
        self.value = config.initial
        self.on_change = on_change

        # Create text visual for display
        self.text = scene.visuals.Text(
            text=self._format_text(),
            pos=(position[0], position[1]),
            color='white',
            font_size=12,
            parent=parent.scene,
        )

    def _format_text(self) -> str:
        """Format the slider display text."""
        bar_length = 20
        normalized = (self.value - self.config.min_val) / (self.config.max_val - self.config.min_val)
        filled = int(normalized * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        val_str = self.config.format_str.format(self.value)
        return f"{self.config.name}: {bar} {val_str}"

    def set_value(self, value: float) -> None:
        """Update slider value."""
        self.value = np.clip(value, self.config.min_val, self.config.max_val)
        self.text.text = self._format_text()
        if self.on_change:
            self.on_change(self.value)

    def update_display(self) -> None:
        """Refresh the display text."""
        self.text.text = self._format_text()
