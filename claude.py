"""HDR accumulation renderer for pre-computed ray trees."""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from raytracer.rays import RayTree


class HDRRayRenderer:
    """Accumulates ray intensities in HDR buffer and applies tone mapping."""
    
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        x_lims: tuple[float, float] = (-8.0, 1.0),
        y_lims: tuple[float, float] = (-4.0, 4.0),
    ):
        self.width = width
        self.height = height
        self.x_lims = x_lims
        self.y_lims = y_lims
        
        # HDR buffer - float32, can exceed 1.0!
        self.hdr_buffer = np.zeros((height, width, 3), dtype=np.float32)
        
    def clear(self):
        """Clear the HDR buffer."""
        self.hdr_buffer.fill(0.0)
        
    def world_to_pixel(self, point: np.ndarray) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        x, y = point[0], point[1]
        px = int((x - self.x_lims[0]) / (self.x_lims[1] - self.x_lims[0]) * self.width)
        py = int((self.y_lims[1] - y) / (self.y_lims[1] - self.y_lims[0]) * self.height)
        return px, py
    
    def accumulate_ray_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        intensity: float = 1.0,
        blur_radius: float = 1.5,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Accumulate a ray segment into HDR buffer using Bresenham's algorithm
        with Gaussian splatting for smooth caustics.
        """
        x0, y0 = self.world_to_pixel(start)
        x1, y1 = self.world_to_pixel(end)
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            # Splat with Gaussian kernel for smooth caustics
            self._splat_pixel(x, y, intensity, blur_radius, color)
            
            if x == x1 and y == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def _splat_pixel(
        self,
        cx: int,
        cy: int,
        intensity: float,
        blur_radius: float,
        color: tuple[float, float, float],
    ):
        """Splat intensity with Gaussian kernel around pixel."""
        r = int(np.ceil(blur_radius * 2))
        
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px = cx + dx
                py = cy + dy
                
                if 0 <= px < self.width and 0 <= py < self.height:
                    dist = np.sqrt(dx * dx + dy * dy)
                    weight = np.exp(-dist * dist / (2 * blur_radius * blur_radius))
                    
                    # ADDITIVE ACCUMULATION - NO CLAMPING!
                    self.hdr_buffer[py, px, 0] += intensity * weight * color[0]
                    self.hdr_buffer[py, px, 1] += intensity * weight * color[1]
                    self.hdr_buffer[py, px, 2] += intensity * weight * color[2]
    
    def draw_ray_tree(
        self,
        tree: RayTree,
        intensity_per_ray: float = 0.001,
        blur_radius: float = 1.5,
        max_length: float = 100.0,
    ):
        """Draw entire ray tree into HDR buffer."""
        for label, node in tree.nodes.items():
            start = node.ray.origin
            
            if node.intersection is not None:
                end = node.intersection.point
            else:
                # Ray doesn't hit - extend in direction
                end = start + node.ray.direction * max_length
            
            # Different colors for different generations (optional)
            generation_colors = [
                (1.0, 1.0, 1.0),  # white
                (1.0, 0.9, 0.8),  # warm white
                (0.9, 0.95, 1.0), # cool white
                (1.0, 0.95, 0.9), # cream
            ]
            gen_idx = min(node.generation - 1, len(generation_colors) - 1)
            color = generation_colors[gen_idx]
            
            self.accumulate_ray_segment(
                start, end,
                intensity=intensity_per_ray,
                blur_radius=blur_radius,
                color=color,
            )
    
    def tone_map(
        self,
        exposure: float = 1.0,
        mode: str = "reinhard",
        gamma: float = 2.2,
    ) -> np.ndarray:
        """
        Apply tone mapping to HDR buffer.
        
        Args:
            exposure: Exposure multiplier
            mode: "reinhard", "filmic", "logarithmic", or "linear"
            gamma: Gamma correction value
            
        Returns:
            8-bit RGB image ready for display
        """
        # Apply exposure
        hdr = self.hdr_buffer * exposure
        
        # Tone mapping
        if mode == "reinhard":
            mapped = hdr / (1.0 + hdr)
        elif mode == "filmic":
            x = np.maximum(0, hdr - 0.004)
            mapped = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        elif mode == "logarithmic":
            luminance = 0.2126 * hdr[:,:,0] + 0.7152 * hdr[:,:,1] + 0.0722 * hdr[:,:,2]
            luminance = np.maximum(luminance, 1e-6)
            log_lum = np.log(1.0 + luminance * 10.0) / np.log(1.0 + 10.0)
            scale = log_lum / luminance
            mapped = hdr * scale[:, :, np.newaxis]
        else:  # linear
            mapped = np.clip(hdr, 0, 1)
        
        # Gamma correction
        mapped = np.power(np.clip(mapped, 0, 1), 1.0 / gamma)
        
        # Convert to 8-bit
        return (mapped * 255).astype(np.uint8)
    
    def show(
        self,
        exposure: float = 1.0,
        mode: str = "reinhard",
        title: str = "HDR Ray Caustics",
        cmap: str = "hot",
    ):
        """Display the tone-mapped result."""
        img = self.tone_map(exposure=exposure, mode=mode)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img, extent=[self.x_lims[0], self.x_lims[1], 
                                self.y_lims[0], self.y_lims[1]],
                   origin='upper', interpolation='bilinear')
        plt.title(f"{title} | Exposure: {exposure:.2f} | Mode: {mode}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Intensity")
        plt.tight_layout()
        plt.show()
    
    def save(self, filename: str, exposure: float = 1.0, mode: str = "reinhard"):
        """Save the tone-mapped result to file."""
        img = self.tone_map(exposure=exposure, mode=mode)
        plt.imsave(filename, img)
        print(f"Saved HDR render to {filename}")


# Example integration with your existing code
def render_ellipse_caustics_hdr():
    """Modified version of your main() using HDR rendering."""
    from pathlib import Path
    import sys
    
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    from raytracer.geometry import EllipseConic
    from raytracer.sources import PointSource2D
    from raytracer.tracer import RayTracer2D, TraceConfig
    
    # Same setup as your original code
    semi_major = 4.0
    semi_minor = 2.5
    ellipse = EllipseConic(
        semi_major=semi_major,
        semi_minor=semi_minor,
        focus=np.array([0.0, 0.0]),
        surface_id="mirror",
    )
    
    primary_focus = ellipse.focus.copy()
    major_axis_dir = ellipse.frame.direction_to_world(np.array([1.0, 0.0]))
    major_axis_dir = major_axis_dir / np.linalg.norm(major_axis_dir)
    focal_offset = np.sqrt(semi_major ** 2 - semi_minor ** 2)
    
    source = PointSource2D(
        origin=np.array([0.0, 0.11]),
        axis_direction=-major_axis_dir,
        aperture=np.deg2rad(80.0),
        samples=5000,
    )
    
    config = TraceConfig(max_generations=3, allow_reflection=True, allow_refraction=False)
    tracer = RayTracer2D(surfaces=[ellipse], config=config)
    tree = tracer.trace([source])
    
    # HDR RENDERING instead of VisPy alpha blending
    renderer = HDRRayRenderer(
        width=1920,
        height=1080,
        x_lims=(-8.0, 1.0),
        y_lims=(-4.0, 4.0),
    )
    
    # Calculate intensity per ray
    intensity_per_ray = 1.0 / source.samples  # Normalized
    
    # Draw all rays into HDR buffer
    renderer.draw_ray_tree(
        tree,
        intensity_per_ray=intensity_per_ray * 50,  # Boost for visibility
        blur_radius=1.5,  # Adjust for smoothness
    )
    
    # Try different visualizations
    renderer.show(exposure=1.5, mode="reinhard", title="Reinhard Tone Mapping")
    renderer.show(exposure=2.0, mode="filmic", title="Filmic Tone Mapping")
    renderer.show(exposure=1.0, mode="logarithmic", title="Logarithmic Tone Mapping")
    
    # Save the best one
    renderer.save("caustics_hdr.png", exposure=1.5, mode="reinhard")


if __name__ == "__main__":
    render_ellipse_caustics_hdr()
