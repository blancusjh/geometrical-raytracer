# geometries.py 

from numpy import sin, cos, pi 
import numpy as np 

from vispy import app, scene 

def setup_canvas(x_lims = (-2, 2), y_lims = (-2, 2), show_axis = True ):     

    canvas = scene.SceneCanvas(keys='interactive', show = True, bgcolor='black', 
                               size = (800, 600))

    view = canvas.central_widget.add_view() 
    view.camera = scene.cameras.PanZoomCamera(aspect=1) 

    if show_axis == True: 
        scene.visuals.XYZAxis(parent=view.scene)

        X0, X1 = x_lims
        Y0, Y1 = y_lims
        view.camera.set_range(x=(X0, X1) , y=(Y0, Y1))
    
    return canvas, view

canvas, view = setup_canvas() 


# Begin

def circle(theta, R = 1.0):     
    x = R*cos(theta) 
    y = R*sin(theta)
 
    return x, y 


def conic(theta, e=0.5, p=1.0):
    """
    General conic in polar coordinates with focus at origin.

    r(θ) = p / (1 + e*cos(θ))

    Parameters:
        theta : ndarray of angles
        e     : eccentricity
        p     : semi-latus rectum (scale)

    Returns:
        x, y : ndarray, Cartesian coordinates
    """
    r = p / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y



N = 60
theta = np.linspace(0, 2*pi, N)
x_conic, y_conic = conic(theta, e=1.0, p=1.0) 
xx_conic = np.array([x_conic, y_conic]).T


positions_conic = xx_conic



# Showing Visuals.

scene.visuals.Line(pos=positions_conic, width=3, color='white', parent=view.scene)

if __name__ == '__main__': 
    app.run()




















