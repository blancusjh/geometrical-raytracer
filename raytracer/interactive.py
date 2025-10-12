"""Interactive controls for ray tracing visualization."""

from __future__ import annotations

from typing import Callable, Any, Sequence, Mapping
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
        """Initialise controller with keyboard-driven sliders."""

        self.canvas = canvas
        self.update_callback = update_callback
        self.sliders: dict[str, SliderConfig] = {}
        self.values: dict[str, float] = {}
        self.widgets: list[Any] = []

        self._active_param: str | None = None
        self._param_list: list[str] = []
        self._numeric_entry: dict[str, Any] | None = None
        self._overlay_params: set[str] = set()
        self._overlay: NumericOverlay | None = None

        self._setup_keyboard_controls()

    def add_slider(self, config: SliderConfig) -> None:
        """Add a slider control."""

        self.values[config.name] = config.initial
        self.sliders[config.name] = config
        self._param_list.append(config.name)
        if self._active_param is None:
            self._active_param = config.name
        self._refresh_overlay()

    def _setup_keyboard_controls(self) -> None:
        """Setup keyboard controls for slider adjustment."""

        @self.canvas.events.key_press.connect
        def on_key(event):
            if self._numeric_entry is not None:
                self._handle_numeric_entry(event)
                return

            if self._active_param is None:
                return

            config = self.sliders.get(self._active_param)
            if config is None:
                return

            current_val = self.values[self._active_param]
            base_step = config.step or (config.max_val - config.min_val) / 100.0
            step = base_step
            modifiers = {m for m in (event.modifiers or ())}
            if 'Shift' in modifiers:
                step *= 10.0
            if 'Control' in modifiers or 'Ctrl' in modifiers:
                step *= 0.1
            if 'Alt' in modifiers:
                step *= 0.01

            changed = False

            if event.key in ('Up', 'Right'):
                new_val = self._sanitize_value(config, current_val + step)
                if new_val != current_val:
                    self.values[self._active_param] = new_val
                    changed = True
            elif event.key in ('Down', 'Left'):
                new_val = self._sanitize_value(config, current_val - step)
                if new_val != current_val:
                    self.values[self._active_param] = new_val
                    changed = True
            elif event.key == 'Tab':
                if self._param_list:
                    idx = self._param_list.index(self._active_param)
                    if 'Shift' in modifiers:
                        idx = (idx - 1) % len(self._param_list)
                    else:
                        idx = (idx + 1) % len(self._param_list)
                    self._active_param = self._param_list[idx]
                    print(f"Active parameter: {self._active_param} = {self.values[self._active_param]}")
                    self._refresh_overlay()
                return
            elif event.key == ' ':
                changed = True
            elif event.key in ('Enter', 'Return'):
                if self._overlay and self._active_param in self._overlay_params:
                    seed = config.format_str.format(current_val)
                    self._numeric_entry = {
                        'param': self._active_param,
                        'buffer': seed,
                        'replace': True,
                    }
                    self._refresh_overlay()
                return

            if changed:
                self._emit_change(self._active_param, config)
            else:
                self._refresh_overlay()

    def _sanitize_value(self, config: SliderConfig, value: float) -> float:
        clamped = float(np.clip(value, config.min_val, config.max_val))
        if isinstance(config.initial, (int, np.integer)):
            return float(int(round(clamped)))
        return clamped

    def _emit_change(self, name: str, config: SliderConfig) -> None:
        formatted = config.format_str.format(self.values[name])
        print(f"{name} = {formatted}")
        self._refresh_overlay()
        self.update_callback(self.values.copy())

    def _handle_numeric_entry(self, event) -> None:
        if self._numeric_entry is None:
            return

        param = self._numeric_entry['param']
        buffer = self._numeric_entry['buffer']
        config = self.sliders.get(param)
        if config is None:
            self._numeric_entry = None
            self._refresh_overlay()
            return

        key = event.key
        text = event.text or ''

        if key in ('Escape',):
            self._numeric_entry = None
            self._refresh_overlay()
            return

        if key in ('Enter', 'Return'):
            self._numeric_entry = None
            try:
                entered = float(buffer)
            except ValueError:
                print(f"Invalid numeric input for {param!r}: {buffer!r}")
                self._refresh_overlay()
                return
            new_val = self._sanitize_value(config, entered)
            if isinstance(config.initial, (int, np.integer)):
                new_val = float(int(round(new_val)))
            if new_val != self.values[param]:
                self.values[param] = new_val
                self._emit_change(param, config)
            else:
                self._refresh_overlay()
            return

        if key == 'Backspace':
            if self._numeric_entry.get('replace'):
                buffer = ''
            else:
                buffer = buffer[:-1]
            self._numeric_entry['buffer'] = buffer
            self._numeric_entry['replace'] = False
            self._refresh_overlay()
            return

        if key == 'Delete':
            self._numeric_entry['buffer'] = ''
            self._numeric_entry['replace'] = False
            self._refresh_overlay()
            return

        if text:
            if self._numeric_entry.get('replace'):
                buffer = ''
                self._numeric_entry['replace'] = False
            if text in '+-' and not buffer:
                buffer = text
            elif text == '.' and not buffer:
                buffer = '0.'
            elif text.isdigit() or text == '.':
                buffer += text
            else:
                return
            self._numeric_entry['buffer'] = buffer
            self._refresh_overlay()

    def enable_numeric_overlay(
        self,
        params: Sequence[str],
        *,
        format_overrides: Mapping[str, str] | None = None,
        anchor: tuple[float, float] = (15.0, 40.0),
        line_height: float = 18.0,
    ) -> None:
        formats: dict[str, str] = {}
        overrides = format_overrides or {}
        for name in params:
            if name in overrides:
                formats[name] = overrides[name]
            elif name in self.sliders:
                formats[name] = self.sliders[name].format_str
            else:
                formats[name] = "{:.3f}"

        self._overlay_params = set(params)
        self._overlay = NumericOverlay(self.canvas, params, formats, anchor=anchor, line_height=line_height)
        self._refresh_overlay()

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
        print("Shift+Tab : Switch backward")
        print("Up/Right  : Increase value")
        print("Down/Left : Decrease value")
        print("Shift     : Coarse step (×10)")
        print("Ctrl      : Fine step (÷10)")
        print("Alt       : Ultra-fine step (×0.01)")
        print("Enter     : Direct numeric entry (tracked params)")
        print("Esc       : Cancel numeric entry")
        print("Space     : Force update")
        print("\nParameters:")
        for name, config in self.sliders.items():
            marker = "(*)" if name == self._active_param else "   "
            print(f"{marker} {name}: {config.format_str.format(self.values[name])} "
                  f"[{config.min_val}, {config.max_val}]")
        print("===========================\n")

    def _refresh_overlay(self) -> None:
        if not self._overlay:
            return
        editing: tuple[str, str] | None = None
        if self._numeric_entry is not None:
            editing = (self._numeric_entry['param'], self._numeric_entry['buffer'])
        self._overlay.update(self.values, self._active_param, editing)


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


class NumericOverlay:
    """Compact textual overlay that tracks selected parameters."""

    def __init__(
        self,
        canvas: scene.SceneCanvas,
        params: Sequence[str],
        formats: Mapping[str, str],
        *,
        anchor: tuple[float, float] = (15.0, 40.0),
        line_height: float = 18.0,
    ) -> None:
        self.params = list(params)
        self.formats = dict(formats)
        self.text_nodes: dict[str, scene.visuals.Text] = {}

        x0, y0 = anchor
        for idx, name in enumerate(self.params):
            node = scene.visuals.Text(
                text=f"{name}: --",
                pos=(x0, y0 + idx * line_height),
                color='#bbbbbb',
                font_size=12,
                anchor_x='left',
                anchor_y='top',
                parent=canvas.scene,
            )
            self.text_nodes[name] = node

    def update(
        self,
        values: Mapping[str, float],
        active: str | None,
        editing: tuple[str, str] | None,
    ) -> None:
        editing_param = editing[0] if editing else None
        buffer = editing[1] if editing else ""

        for name, node in self.text_nodes.items():
            fmt = self.formats.get(name, "{:.3f}")
            if editing_param == name:
                display = f"{name}: [{buffer}]"
                node.color = '#00d1ff'
            else:
                value = values.get(name, 0.0)
                try:
                    val_str = fmt.format(value)
                except Exception:  # pragma: no cover - defensive
                    val_str = str(value)
                display = f"{name}: {val_str}"
                node.color = 'white' if name == active else '#bbbbbb'
            node.text = display
