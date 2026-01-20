# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "easyocr",
#     "pytesseract",
# ]
# ///
import cv2
import numpy as np
import easyocr


class TextCharacterDetector:
    """Detector especializado para detectar y reconocer texto/caracteres en video usando OCR"""

    def __init__(self, confidence=0.5, ocr_languages=["en", "es"]):
        self.confidence = confidence
        self.ocr_reader = None
        self.ocr_languages = ocr_languages

        # Colores para diferentes tipos de texto
        self.colors = {
            "letters": (0, 255, 0),  # Verde para letras
            "numbers": (255, 0, 0),  # Azul para números
            "mixed": (0, 255, 255),  # Amarillo para texto mixto
            "symbols": (255, 0, 255),  # Magenta para símbolos
        }

    def load_models(self):
        """Inicializar OCR"""
        print("Inicializando EasyOCR...")
        self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=False)
        print("✓ EasyOCR inicializado")

    def classify_text_type(self, text: str) -> str:
        """Clasificar el tipo de texto detectado"""
        text = text.strip()
        if not text:
            return "unknown"

        # Solo letras
        if text.isalpha():
            return "letters"
        # Solo números
        elif text.isdigit():
            return "numbers"
        # Contiene letras y números
        elif any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
            return "mixed"
        # Otros caracteres (símbolos, puntuación, etc.)
        else:
            return "symbols"

    def detect_text_regions(self, frame):
        """Detectar regiones que pueden contener texto usando procesamiento de imagen"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar filtros para resaltar texto
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detectar bordes
        edges = cv2.Canny(blurred, 50, 150)

        # Dilatación para conectar caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Encontrar contornos
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filtrar regiones muy pequeñas o con aspect ratio extraño
            if (
                w > 15
                and h > 15
                and w < frame.shape[1] * 0.8
                and h < frame.shape[0] * 0.8
            ):
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:  # Filtrar formas muy extrañas
                    text_regions.append((x, y, x + w, y + h))

        return text_regions

    def extract_text_from_regions(self, frame, regions):
        """Extraer texto de las regiones detectadas usando OCR"""
        detected_texts = []

        for x1, y1, x2, y2 in regions:
            # Extraer región de interés
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            try:
                # Usar EasyOCR para extraer texto
                results = self.ocr_reader.readtext(roi, detail=1)

                for bbox_coords, text, confidence in results:
                    if confidence > 0.5:  # Filtro de confianza para OCR
                        # Ajustar coordenadas al frame completo
                        text_type = self.classify_text_type(text)

                        detected_texts.append(
                            {
                                "text": text,
                                "bbox": (x1, y1, x2, y2),
                                "confidence": confidence,
                                "type": text_type,
                                "roi_bbox": bbox_coords,
                            }
                        )

            except Exception:
                # Si hay error con OCR, continúa con la siguiente región
                continue

        return detected_texts

    def detect_characters(self, frame):
        """Detectar y reconocer caracteres/texto en el frame"""
        if self.ocr_reader is None:
            raise ValueError("OCR no inicializado. Llama a load_models() primero.")

        # Método 1: Detectar regiones de texto usando procesamiento de imagen
        text_regions = self.detect_text_regions(frame)

        # Método 2: Extraer texto usando OCR
        detected_texts = self.extract_text_from_regions(frame, text_regions)

        return detected_texts

    def draw_detections(self, frame, detected_texts):
        """Dibujar las detecciones de texto en el frame"""
        annotated_frame = frame.copy()

        for i, detection in enumerate(detected_texts):
            x1, y1, x2, y2 = detection["bbox"]
            text = detection["text"]
            confidence = detection["confidence"]
            text_type = detection["type"]

            # Color según el tipo de texto
            color = self.colors.get(text_type, (255, 255, 255))

            # Dibujar rectángulo
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Preparar etiqueta
            label = f"{text} ({text_type}) {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Fondo para el texto
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1,
            )

            # Texto
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Información general
        total_text = len(detected_texts)
        cv2.putText(
            annotated_frame,
            f"Texto detectado: {total_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return annotated_frame




def main():
    """Función principal para ejecutar la detección de caracteres en tiempo real"""
    print("=== DETECCIÓN DE CARACTERES/TEXTO EN TIEMPO REAL ===")
    print("Presiona 'q' para salir")
    print("Presiona 'p' para pausar/reanudar\n")

    # Inicializar detector
    detector = TextCharacterDetector(confidence=0.5)
    detector.load_models()

    # Configurar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("✓ Cámara configurada")
    print("✓ Sistema listo para detectar texto")

    # Crear ventana
    window_name = "Detección de Caracteres/Texto - OCR"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer el frame de la cámara.")
                break

            frame_count += 1

            # Detectar texto/caracteres
            detected_texts = detector.detect_characters(frame)

            # Dibujar detecciones
            annotated_frame = detector.draw_detections(frame, detected_texts)
        else:
            # Si está pausado, mantener el último frame
            pass

        # Mostrar frame
        cv2.imshow(window_name, annotated_frame)

        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'Pausado' if paused else 'Reanudado'}")

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Programa terminado. Frames procesados: {frame_count}")


if __name__ == "__main__":
    main()
