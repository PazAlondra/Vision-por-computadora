import cv2
import numpy

# SciPy da interpolacion cubica; si no esta instalado, se usa NumPy.
try:
    import scipy.interpolate
except ImportError:
    scipy = None

def createCurveFunc(points):
    """Return a function derived from control points."""
    # Sin puntos no hay curva que crear.
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    # Con pocos puntos se usa linea; con mas puntos se intenta curva cubica.
    if numPoints < 4:
        kind = 'linear'
        # 'quadratic' is not implemented.
    else:
        kind = 'cubic'
    # Usa SciPy si esta disponible; si no, NumPy interpola linealmente.
    if scipy is not None:
        return scipy.interpolate.interp1d(xs, ys, kind,
                                          bounds_error = False)
    return lambda x: numpy.interp(x, xs, ys)
    
def createLookupArray(func, length = 256):
    """Return a lookup for whole-number inputs to a function.
    The lookup values are clamped to [0, length - 1].
    """
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    # Calcula una salida para cada valor posible de entrada.
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray
def applyLookupArray(lookupArray, src, dst):
    """Map a source to a destination using a lookup."""
    if lookupArray is None:
        return
    # Usa los valores del arreglo fuente como indices de la tabla.
    dst[:] = lookupArray[src]

def createCompositeFunc(func0, func1):
    """Return a composite of two functions."""
    # Si falta una funcion, regresa la otra directamente.
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    # Aplica primero func1 y luego func0.
    return lambda x: func0(func1(x))

def createFlatView(array):
    """Return a 1D view of an array of any dimensionality."""
    # Crea una vista plana sin copiar los datos.
    flatView = array.view()
    flatView.shape = array.size
    return flatView

# Alias usado por filters.py.
flatView = createFlatView


