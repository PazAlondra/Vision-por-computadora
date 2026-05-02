import cv2
import numpy as np
import os
import time


class CaptureManager(object):
    """Administra la camara, los frames, las fotos y la grabacion de video."""

    def __init__(self, capture, previewWindowManager=None,
                 shouldMirrorPreview=False, previewCallback=None):
        # Ventana donde se muestra el preview de la camara.
        self.previewWindowManager = previewWindowManager
        # Si es True, muestra el preview como espejo.
        self.shouldMirrorPreview = shouldMirrorPreview
        # Funcion opcional para dibujar encima del frame antes de mostrarlo.
        self.previewCallback = previewCallback
        # Objeto cv2.VideoCapture usado para leer la camara.
        self._capture = capture
        self._channel = 0
        # Indica si se entro a un frame que todavia no fue cerrado.
        self._enteredFrame = False
        # Frame actual leido desde la camara.
        self._frame = None
        # Cuando tiene valor, el siguiente frame se guarda como imagen.
        self._imageFilename = None
        # Cuando tiene valor, los frames se escriben a video.
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Capture the next frame, if any."""
        # Evita entrar dos veces al mismo ciclo sin llamar exitFrame().
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            # read() devuelve si pudo capturar y el frame capturado.
            grabbed, frame = self._capture.read()
            self._enteredFrame = grabbed
            if grabbed:
                self._frame = frame
            else:
                self._frame = None

    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""
        # Si no hay frame valido, solo limpia el estado de entrada.
        if self._frame is None:
            self._enteredFrame = False
            return

        # Calcula una estimacion de FPS para crear videos cuando la camara
        # no reporta un FPS confiable.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            if timeElapsed > 0:
                self._fpsEstimate = self._framesElapsed / timeElapsed

        self._framesElapsed += 1

        # Muestra el preview; si hay callback, primero dibuja la interfaz.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                if self.previewCallback is not None:
                    self.previewCallback(mirroredFrame)
                self.previewWindowManager.show(mirroredFrame)
            else:
                if self.previewCallback is not None:
                    self.previewCallback(self._frame)
                self.previewWindowManager.show(self._frame)

        if self.isWritingImage:
            # Guarda una sola imagen y luego limpia la solicitud.
            if not cv2.imwrite(self._imageFilename, self._frame):
                print('No se pudo guardar la foto: %s' % self._imageFilename)
            else:
                print('Foto guardada: %s' % self._imageFilename)
            self._imageFilename = None

        self._writeVideoFrame()

        # Libera la referencia al frame para el siguiente ciclo.
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        # Crea la carpeta destino antes de solicitar el guardado.
        self._createParentFolder(filename)
        self._imageFilename = filename

    def startWritingVideo(self, filename,
                          encoding=cv2.VideoWriter_fourcc(*'XVID')):
        """Start writing exited frames to a video file."""
        # Solo prepara el nombre y codec; el VideoWriter se abre al primer
        # frame valido porque ahi ya se conocen FPS y tamano.
        self._createParentFolder(filename)
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frames to a video file."""
        # release() termina correctamente el archivo de video.
        if self._videoWriter is not None:
            self._videoWriter.release()
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def release(self):
        """Release camera and video-writing resources."""
        # Cierra primero el video, luego la camara.
        self.stopWritingVideo()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._frame = None
        self._enteredFrame = False

    def _writeVideoFrame(self):
        # Si no hay grabacion activa, no escribe nada.
        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            # Intenta usar el FPS real de la camara.
            fps = self._capture.get(cv2.CAP_PROP_FPS)

            if fps <= 0:
                # Espera algunos frames para calcular FPS aproximado.
                if self._framesElapsed < 20:
                    return
                fps = self._fpsEstimate if self._fpsEstimate is not None else 30.0

            # Toma el tamano del frame desde la camara.
            size = (
                int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            # Respaldo por si la camara no reporta dimensiones validas.
            if size[0] <= 0 or size[1] <= 0:
                size = (640, 480)

            # Crea el escritor de video con archivo, codec, FPS y tamano.
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename,
                self._videoEncoding,
                fps,
                size
            )
            if not self._videoWriter.isOpened():
                print('No se pudo iniciar el video: %s' % self._videoFilename)
                self.stopWritingVideo()
                return
            print('Grabando video: %s' % self._videoFilename)

        # Escribe el frame actual en el archivo de video.
        self._videoWriter.write(self._frame)

    def _createParentFolder(self, filename):
        """Crea la carpeta padre del archivo si no existe."""
        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)


class WindowManager(object):
    """Administra la ventana de OpenCV y sus eventos."""

    def __init__(self, windowName, keypressCallback=None,
                 mouseCallback=None):
        # Callbacks opcionales para teclado y mouse.
        self.keypressCallback = keypressCallback
        self.mouseCallback = mouseCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        # Crea la ventana y conecta el mouse si hay callback configurado.
        cv2.namedWindow(self._windowName)
        if self.mouseCallback is not None:
            cv2.setMouseCallback(self._windowName, self.mouseCallback)
        self._isWindowCreated = True

    def show(self, frame):
        # Muestra el frame actual en la ventana.
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        # Evita intentar cerrar dos veces la misma ventana.
        if not self._isWindowCreated:
            return
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        # waitKey permite que OpenCV procese eventos de ventana y teclado.
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)
