import cv2
import numpy
import utils


# Filtros de recolor: modifican canales BGR directamente.
def recolorRC(src, dst):
    """Simulate conversion from BGR to RC (red, cyan).
    The source and destination images must both be in BGR format.
    Blues and greens are replaced with cyans.
    Pseudocode:
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    # Mezcla azul y verde para crear un canal cian.
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    cv2.merge((b, b, r), dst)


def recolorRGV(src, dst):
    """Simulate conversion from BGR to RGV (red, green, value).
    The source and destination images must both be in BGR format.
    Blues are desaturated.
    Pseudocode:
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    # El canal azul queda como el minimo entre azul, verde y rojo.
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    cv2.merge((b, g, r), dst)


def recolorCMV(src, dst):
    """Simulate conversion from BGR to CMV (cyan, magenta, value).
    The source and destination images must both be in BGR format.
    Yellows are desaturated.
    Pseudocode:
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b, g, r = cv2.split(src)
    # El canal azul queda como el maximo entre azul, verde y rojo.
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)."""

    def __init__(self, vFunc=None, dtype=numpy.uint8):
        # Convierte la funcion en una tabla de busqueda para aplicar rapido.
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        srcFlatView = utils.flatView(src)
        dstFlatView = utils.flatView(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)


class VCurveFilter(VFuncFilter):
    """A filter that applies a curve to V (or all of BGR)."""

    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)


class BGRFuncFilter(object):
    """A filter that applies different functions to each of BGR."""

    def __init__(self, vFunc=None, bFunc=None, gFunc=None,
                 rFunc=None, dtype=numpy.uint8):
        # Crea una tabla de busqueda independiente para cada canal BGR.
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(
            utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """Apply the filter with a BGR source/destination."""
        # Separa canales, transforma cada uno y vuelve a unirlos.
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curves to each of BGR."""

    def __init__(self, vPoints=None, bPoints=None,
                 gPoints=None, rPoints=None, dtype=numpy.uint8):
        # Convierte puntos de control en curvas para brillo y canales.
        BGRFuncFilter.__init__(
            self,
            utils.createCurveFunc(vPoints),
            utils.createCurveFunc(bPoints),
            utils.createCurveFunc(gPoints),
            utils.createCurveFunc(rPoints),
            dtype
        )


class BGRPortraCurveFilter(BGRCurveFilter):
    """A filter that applies Portra-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        # Curvas con aspecto tipo pelicula Portra.
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype=dtype
        )


class BGRProviaCurveFilter(BGRCurveFilter):
    """A filter that applies Provia-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        # Curvas con aspecto tipo pelicula Provia.
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
            gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
            rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
            dtype=dtype
        )


class BGRVelviaCurveFilter(BGRCurveFilter):
    """A filter that applies Velvia-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        # Curvas con colores mas intensos tipo Velvia.
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
            bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
            dtype=dtype
        )


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """A filter that applies cross-process-like curves to BGR."""

    def __init__(self, dtype=numpy.uint8):
        # Curvas para simular cross-process.
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 20), (255, 235)],
            gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
            rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
            dtype=dtype
        )


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    """Dibuja bordes oscureciendo zonas detectadas por Laplacian."""
    if blurKsize >= 3:
        # Suaviza antes de detectar bordes para reducir ruido.
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # CORREGIDO: antes decía cv2.cv.CV_8U
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)

    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)

    for channel in channels:
        # Aplica los bordes como mascara a cada canal.
        channel[:] = channel * normalizedInverseAlpha

    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""

    def __init__(self, kernel):
        # Kernel usado por OpenCV para transformar la imagen.
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        # filter2D aplica la matriz de convolucion al frame.
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius."""

    def __init__(self):
        # Kernel que aumenta contraste local para dar nitidez.
        kernel = numpy.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""

    def __init__(self):
        # Kernel que resalta cambios fuertes entre pixeles.
        kernel = numpy.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        VConvolutionFilter.__init__(self, kernel)


class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        # Kernel que crea un efecto de relieve.
        kernel = numpy.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [0,   1, 2]
        ])
        VConvolutionFilter.__init__(self, kernel)
