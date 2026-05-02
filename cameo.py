import os
import cv2
import filters
from datetime import datetime
from managers import WindowManager, CaptureManager


class Cameo(object):
    """Clase principal: abre la camara, aplica filtros y maneja la interfaz."""

    def __init__(self):
        # Crea la ventana principal y registra callbacks de teclado y mouse.
        self._windowManager = WindowManager(
            'Cameo', self.onKeypress, self.onMouse
        )

        # Abre la camara predeterminada. CAP_DSHOW ayuda en Windows.
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # Administra captura, preview, espejo horizontal y dibujo de botones.
        self._captureManager = CaptureManager(
            camera, self._windowManager, True, self._drawInterface
        )

        # Lista de filtros disponibles. El primer valor es el texto del boton.
        # El segundo valor es una funcion, un objeto con apply(), o None.
        self._filters = [
            ('NORMAL', None),
            ('RC', filters.recolorRC),
            ('RGV', filters.recolorRGV),
            ('CMV', filters.recolorCMV),
            ('PORTRA', filters.BGRPortraCurveFilter()),
            ('PROVIA', filters.BGRProviaCurveFilter()),
            ('VELVIA', filters.BGRVelviaCurveFilter()),
            ('CROSS', filters.BGRCrossProcessCurveFilter()),
            ('SHARP', filters.SharpenFilter()),
            ('EDGES', filters.FindEdgesFilter()),
            ('EMBOSS', filters.EmbossFilter())
        ]
        # Filtro activo al iniciar el programa.
        self._filterName = 'PORTRA'
        # Guarda las coordenadas de cada boton para detectar clics.
        self._buttonRects = []
        # Asegura que exista la carpeta base.
        os.makedirs("CAMERA", exist_ok=True)

    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()

        try:
            # Ciclo principal: captura frame, aplica filtro, muestra preview
            # y procesa teclado/mouse hasta que se cierre la ventana.
            while self._windowManager.isWindowCreated:
                self._captureManager.enterFrame()
                frame = self._captureManager.frame

                if frame is not None:
                    self._applySelectedFilter(frame)

                self._captureManager.exitFrame()
                self._windowManager.processEvents()
        finally:
            # Aunque ocurra un error, libera camara, video y ventana.
            self._captureManager.release()
            self._windowManager.destroyWindow()

    def onKeypress(self, keycode):
        """Maneja atajos de teclado."""
        if keycode == 32:  # espacio
            self._takePhoto()
        elif keycode == 9:  # tab
            self._toggleVideo()
        elif keycode == 27:  # esc
            self._captureManager.release()
            self._windowManager.destroyWindow()

    def onMouse(self, event, x, y, flags, param):
        """Maneja clics sobre los botones dibujados en la imagen."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Revisa si el clic quedo dentro de algun rectangulo de boton.
        for label, action, x0, y0, x1, y1 in self._buttonRects:
            if x0 <= x <= x1 and y0 <= y <= y1:
                if action == 'photo':
                    self._takePhoto()
                elif action == 'video':
                    self._toggleVideo()
                elif action == 'filter':
                    self._filterName = label
                break

    def _applySelectedFilter(self, frame):
        """Busca el filtro seleccionado y lo aplica al frame actual."""
        for name, frameFilter in self._filters:
            if name != self._filterName:
                continue
            if frameFilter is None:
                return
            if callable(frameFilter):
                frameFilter(frame, frame)
            else:
                frameFilter.apply(frame, frame)
            return

    def _takePhoto(self):
        """Solicita guardar el siguiente frame como imagen."""
        filename = 'img/photo_%s.png' % self._timestamp()
        self._captureManager.writeImage(filename)

    def _toggleVideo(self):
        """Inicia o detiene la grabacion de video."""
        if not self._captureManager.isWritingVideo:
            filename = 'video/video_%s.mp4' % self._timestamp()
            encoding = cv2.VideoWriter_fourcc(*'mp4v')
            self._captureManager.startWritingVideo(filename, encoding)
        else:
            self._captureManager.stopWritingVideo()

    def _timestamp(self):
        """Genera una marca de fecha/hora para nombres de archivos."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    def _drawInterface(self, frame):
        """Dibuja la barra de botones encima del frame de la camara."""
        self._buttonRects = []

        # Medidas generales de la barra y los botones.
        height, width = frame.shape[:2]
        buttonHeight = 32
        padding = 8
        gap = 6
        x = padding
        y = padding

        # Fondo oscuro semitransparente para que se lean los botones.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 92), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Botones de acciones principales y un boton por cada filtro.
        controls = [
            ('FOTO', 'photo'),
            ('VIDEO', 'video')
        ]
        controls.extend((name, 'filter') for name, unused in self._filters)

        for label, action in controls:
            # Calcula el ancho del boton segun el texto.
            textSize, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            buttonWidth = max(74, textSize[0] + 18)

            # Si ya no cabe en la fila, baja a la siguiente.
            if x + buttonWidth > width - padding:
                x = padding
                y += buttonHeight + gap

            x1 = x + buttonWidth
            y1 = y + buttonHeight
            # Marca como activo el filtro seleccionado o el video grabando.
            isActive = (
                (action == 'video' and self._captureManager.isWritingVideo) or
                (action == 'filter' and label == self._filterName)
            )
            fillColor = (42, 120, 235) if isActive else (58, 58, 58)
            borderColor = (255, 255, 255) if isActive else (120, 120, 120)

            cv2.rectangle(frame, (x, y), (x1, y1), fillColor, -1)
            cv2.rectangle(frame, (x, y), (x1, y1), borderColor, 1)

            # Centra el texto dentro del boton.
            textX = x + (buttonWidth - textSize[0]) // 2
            textY = y + (buttonHeight + textSize[1]) // 2 - baseline
            cv2.putText(
                frame, label, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA
            )
            # Guarda coordenadas para saber que boton fue presionado.
            self._buttonRects.append((label, action, x, y, x1, y1))
            x = x1 + gap


if __name__ == "__main__":
    # Punto de entrada cuando se ejecuta: python cameo.py
    Cameo().run()
