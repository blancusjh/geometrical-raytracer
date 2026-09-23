"""Native VTK view of transparent optical volumes and traced ray paths.

Install the optional ``solid`` extra. Depth peeling resolves overlapping
transparent faces; display opacity is not an optical absorption coefficient.
"""

from pathlib import Path

import numpy as np

from .solid_geometry import glass_blue, lens_meshes


class SolidViewer:
    """Orbitable, orthographic lens view with the same scene used for PNG export."""

    def __init__(self, system, *, size=(1600, 900), opacity=0.42, title=None):
        if not 0 < opacity < 1:
            raise ValueError("glass opacity must lie strictly between zero and one")
        try:
            import vtk
        except ImportError as error:
            raise ImportError("SolidViewer requires pip install 'raytracer[solid]'") from error
        self.vtk = vtk
        self.system = system
        self.meshes = lens_meshes(system)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.055, 0.075, 0.11)
        self.renderer.SetBackground2(0.19, 0.23, 0.29)
        self.renderer.GradientBackgroundOn()
        self.renderer.SetUseDepthPeeling(True)
        self.renderer.SetUseFXAA(True)
        self.renderer.SetMaximumNumberOfPeels(100)
        self.renderer.SetOcclusionRatio(0.0)
        self.window = vtk.vtkRenderWindow()
        self.window.SetSize(*size)
        self.window.SetWindowName(title or system.name or "Geometrical raytracer")
        self.window.SetAlphaBitPlanes(1)
        self.window.SetMultiSamples(0)
        self.window.AddRenderer(self.renderer)
        self.bounds = []
        self._label(title or system.name or "Geometrical raytracer", 0.045, 0.925, 28)
        self._label(
            f"GEOMETRIA REAL  /  mm  /  {system.wavelength_um * 1000:.2f} nm",
            0.046,
            0.88,
            15,
            color=(0.66, 0.75, 0.85),
        )
        legend = []
        for mesh in self.meshes:
            poly = self._polydata(mesh.vertices, triangles=mesh.triangles)
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(poly)
            normals.SetFeatureAngle(45)
            normals.SplittingOn()
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*glass_blue(mesh.refractive_index))
            prop.SetOpacity(opacity)
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.4)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.45)
            prop.SetSpecularPower(55)
            self.renderer.AddActor(actor)
            self.bounds.append(mesh.vertices)
            for rim in mesh.rims:
                self._line(rim, color=glass_blue(mesh.refractive_index), width=1.3, opacity=0.7)
            item = (mesh.material, mesh.refractive_index)
            if item not in legend:
                legend.append(item)
        for index, (name, value) in enumerate(legend):
            self._label(
                f"{name}   n = {value:.5f}", 0.05 + index * 0.24, 0.07, 18, color=glass_blue(value)
            )
        self._label(
            "Azul mas oscuro: mayor indice  |  Transparencia de visualizacion",
            0.05,
            0.025,
            14,
            color=(0.68, 0.76, 0.85),
        )
        self._add_bare_surfaces()

    def _polydata(self, vertices, *, triangles=None, line=False):
        vtk = self.vtk
        points = vtk.vtkPoints()
        for point in vertices:
            points.InsertNextPoint(*point)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        cells = vtk.vtkCellArray()
        if line:
            cells.InsertNextCell(len(vertices))
            for index in range(len(vertices)):
                cells.InsertCellPoint(index)
            poly.SetLines(cells)
        else:
            for triangle in triangles:
                cells.InsertNextCell(3)
                for index in triangle:
                    cells.InsertCellPoint(int(index))
            poly.SetPolys(cells)
        return poly

    def _line(self, points, *, color, width=1, opacity=1):
        # Preserve gaps: joining across a failed ray intersection fabricates paths.
        finite = np.all(np.isfinite(points), axis=1)
        for indices in np.split(np.arange(len(points)), np.flatnonzero(np.diff(finite)) + 1):
            if len(indices) < 2 or not finite[indices[0]]:
                continue
            mapper = self.vtk.vtkPolyDataMapper()
            mapper.SetInputData(self._polydata(points[indices], line=True))
            actor = self.vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(*color)
            prop.SetOpacity(opacity)
            prop.SetLineWidth(width)
            prop.LightingOff()
            self.renderer.AddActor(actor)

    def _label(self, text, x, y, size, *, color=(0.92, 0.96, 1)):
        actor = self.vtk.vtkTextActor()
        actor.SetInput(text)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(x, y)
        prop = actor.GetTextProperty()
        prop.SetFontFamilyToArial()
        prop.SetFontSize(size)
        prop.SetColor(*color)
        self.renderer.AddViewProp(actor)

    def _add_bare_surfaces(self):
        from .geometric import _surface_curves

        covered = {i for mesh in self.meshes for i in mesh.surface_indices}
        for index, row in enumerate(self.system.rows):
            if index in covered:
                continue
            for xy in _surface_curves(row, 128)[:1]:
                points = self.system.surface_frame(index).to_world(
                    np.column_stack([xy, row.profile.sag(np.linalg.norm(xy, axis=1))])
                )
                self._line(points, color=(0.63, 0.69, 0.76), width=1.5)
                self.bounds.append(points[np.all(np.isfinite(points), axis=1)])

    def add_paths(self, paths, *, color=(1, 0.70, 0.27), label=None, detail=False):
        """Add actual world-coordinate paths; detail retains inter-surface legs."""
        for path in np.asarray(paths):
            points = path[1:-1] if detail else path
            self._line(points, color=color, width=2.0, opacity=0.95)
            finite = points[np.all(np.isfinite(points), axis=1)]
            if len(finite):
                self.bounds.append(finite)
        if label:
            count = getattr(self, "_ray_labels", 0)
            self._label(label, 0.76, 0.84 - count * 0.035, 16, color=color)
            self._ray_labels = count + 1

    def frame(self, *, side=False):
        points = np.vstack(self.bounds)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) / 2
        diagonal = max(np.linalg.norm(upper - lower), 1)
        camera = self.renderer.GetActiveCamera()
        direction = np.array([-1, 0, 0]) if side else np.array([-1, 0.28, 0.42])
        camera.SetPosition(*(center + diagonal * self.system.frame.direction_to_world(direction)))
        camera.SetFocalPoint(*center)
        camera.SetViewUp(*self.system.frame.direction_to_world([0, 1, 0]))
        camera.ParallelProjectionOn()
        self.renderer.ResetCamera()
        camera.Zoom(0.85)
        self.renderer.ResetCameraClippingRange()

    def save(self, path, *, side=False):
        """Render the current scene without opening a desktop window."""
        self.window.SetOffScreenRendering(1)
        self.frame(side=side)
        self.window.Render()
        capture = self.vtk.vtkWindowToImageFilter()
        capture.SetInput(self.window)
        capture.ReadFrontBufferOff()
        capture.Update()
        writer = self.vtk.vtkPNGWriter()
        writer.SetFileName(str(Path(path)))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        return {"depth_peeling": bool(self.renderer.GetLastRenderingUsedDepthPeeling())}

    def show(self):
        """Open a native orbit/pan/zoom window. Close it to return to Python."""
        self.window.SetOffScreenRendering(0)
        interactor = self.vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(self.window)
        interactor.SetInteractorStyle(self.vtk.vtkInteractorStyleTrackballCamera())
        self.frame()
        interactor.Initialize()
        self.window.Render()
        interactor.Start()

    def close(self):
        self.window.Finalize()
