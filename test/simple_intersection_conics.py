
# simple_conic_ray.py
import numpy as np
from vispy import app, scene


# ----------------------------
# Canvas setup
# ----------------------------
def setup_canvas(x_lims: tuple[float, float] = (-6, 6),
                 y_lims: tuple[float, float] = (-6, 6),
                 show_axis: bool = True):

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black',
                               size=(900, 700))
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.PanZoomCamera(aspect=1)
    if show_axis:
        scene.visuals.XYZAxis(parent=view.scene)
        X0, X1 = x_lims
        Y0, Y1 = y_lims
        view.camera.set_range(x=(X0, X1), y=(Y0, Y1))

    return canvas, view


# ----------------------------
# Convert polar conic (e,p) -> quadratic coefficients A..F
# ----------------------------
def quadratic_coeffs_from_ep(e: float, p: float) -> tuple[float, float, float, float, float, float]:
    A: float = 1.0 - e**2
    B: float = 0.0
    C: float = 1.0
    D: float = 2.0 * e * p
    E: float = 0.0
    F: float = - (p**2)
    coeffs = (A, B, C, D, E, F)
    return coeffs


# ----------------------------
# Class for dioptrics (conics in polar form)
# ----------------------------
class Dioptrique:

    def __init__(self, e: float, p: float):
        self.e = float(e)
        self.p = float(p)

    # Pure formula for the conic in polar coordinates
    def r(self, theta: float) -> float | None:
        denom = 1.0 + self.e * np.cos(theta)

        if denom <= 1e-12:
            return None

        return self.p / denom

    # Construct array of points (for drawing, not for theory)
    def as_points(self, N: int = 800,
                  theta_span: tuple[float, float] = (-np.pi, np.pi),
                  r_max_clip: float = 1e3) -> np.ndarray:

        thetas = np.linspace(theta_span[0], theta_span[1], N)
        r_values = [self.r(theta) for theta in thetas]
        r_values = np.array([val if val is not None else np.nan for val in r_values])
        r_values = np.clip(r_values, -r_max_clip, r_max_clip)

        x = r_values * np.cos(thetas)
        y = r_values * np.sin(thetas)
        pts = np.column_stack((x, y))

        return pts


# ----------------------------
# Ray - conic intersection
# ----------------------------

# Interection of a ray with a conic by means of 
# the analytical solution of implicit equation.

def intersect_ray_conic(
    P: np.ndarray,
    d: np.ndarray,
    coeffs: tuple[float, float, float, float, float, float],
    eps: float = 1e-12
) -> np.ndarray | None:
    """Find the first intersection of a ray with a conic section.
    
    Ray: P + λd (λ ≥ 0)
    Conic: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    """
    x0, y0 = P[:2]
    dx, dy = d[:2]
    A, B, C, D, E, F = coeffs
    
    # Substitute ray equation into conic: aλ² + bλ + c = 0
    a = A*dx**2 + B*dx*dy + C*dy**2
    b = 2*A*x0*dx + B*(x0*dy + y0*dx) + 2*C*y0*dy + D*dx + E*dy
    c = A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F
    
    # Solve for λ ≥ 0
    lam = None
    if abs(a) < eps:  # Linear case
        if abs(b) >= eps and -c / b >= 0:
            lam = -c / b
    else:  # Quadratic case
        disc = max(0, b**2 - 4*a*c)
        sqrt_disc = np.sqrt(disc)
        roots = [(-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)]
        valid_roots = [t for t in roots if t >= -eps]
        if valid_roots:
            lam = min(valid_roots)
    
    if lam is None:
        return None
    
    Q = P + lam * d
    return Q


# ----------------------------
# Utility: draw polyline with vispy
# ----------------------------
def draw_line(pts: np.ndarray, color: str = 'white', width: int = 2): 
    if pts.size != 0:
        scene.visuals.Line(pos=pts, color=color, parent=view.scene, width=width)


# ----------------------------
# Main execution
# ----------------------------
if __name__ == '__main__':
    canvas, view = setup_canvas()

    # Define conic
    e: float = 1.5
    p: float = 2.0
    conic = Dioptrique(e, p)

    # Coefficients for intersection
    coeffs = quadratic_coeffs_from_ep(e, p)

    # Draw conic
    pts = conic.as_points(N=1200, theta_span=(-np.pi, np.pi))
    draw_line(pts, color='white', width=2)

    # Define ray
    P = np.array([-6.0, 1.0]) #Origin of the ray.
    d = np.array([1.0, 0.0]) # Direction of the ray.

    # Intersection
    Q = intersect_ray_conic(P, d, coeffs)
    if Q is not None:
        ray_pts = np.array([P, Q])
        draw_line(ray_pts, color='yellow', width=3)
        scene.visuals.Markers(pos=np.array([Q]), size=10, face_color='red', parent=view.scene)

    app.run()
