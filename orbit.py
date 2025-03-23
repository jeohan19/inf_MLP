import vtk
import time

# Vytvoření nepravidelného objektu: deformovaná koule
sphere = vtk.vtkSphereSource()
sphere.SetPhiResolution(100)
sphere.SetThetaResolution(100)

# Generování Perlinova šumu pro nepravidelnost
perlin_noise = vtk.vtkPerlinNoise()
perlin_noise.SetFrequency(2.0, 1.0, 0.5)
perlin_noise.SetPhase(0.5, 0.5, 0.5)

# Vytvoření implicitního modifikátoru
implicit_mod = vtk.vtkSampleFunction()
implicit_mod.SetImplicitFunction(perlin_noise)
implicit_mod.SetSampleDimensions(100, 100, 100)
implicit_mod.SetModelBounds(-1, 1, -1, 1, -1, 1)

# Použití kontur pro zobrazení povrchu
contour = vtk.vtkContourFilter()
contour.SetInputConnection(implicit_mod.GetOutputPort())
contour.SetValue(0, 0.0)

# Mapování výsledku
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contour.GetOutputPort())
mapper.ScalarVisibilityOff()

# Vytvoření aktoru a nastavení barvy
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.2, 0.6, 1.0)

# Renderer a renderovací okno
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(600, 600)

# Interaktor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Zobrazení scény
render_window.Render()

# Animace - rotace kolem Y
for i in range(360):
    actor.RotateY(1)
    render_window.Render()
    time.sleep(0.01)

interactor.Start()

