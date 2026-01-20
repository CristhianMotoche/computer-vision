# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "ultralytics",
#     "easyocr",
#     "pytesseract",
# ]
# ///
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


class TextCharacterDetector:
    """Detector especializado para detectar y reconocer texto/caracteres en video usando YOLOv8 + OCR"""

    def __init__(
        self, confidence=0.5, model_path="yolov8n.pt", ocr_languages=["en", "es"]
    ):
        self.confidence = confidence
        self.model = None
        self.model_path = model_path
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
        """Cargar modelo YOLOv8 y inicializar OCR"""
        print("Cargando YOLOv8...")
        self.model = YOLO(self.model_path)
        self.model.to("cpu")
        print("✓ Modelo YOLOv8 cargado")

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
        if self.model is None or self.ocr_reader is None:
            raise ValueError("Modelos no cargados. Llama a load_models() primero.")

        # Método 1: Detectar regiones de texto usando procesamiento de imagen
        text_regions = self.detect_text_regions(frame)

        # Método 2: Extraer texto usando OCR
        detected_texts = self.extract_text_from_regions(frame, text_regions)

        return detected_texts

    def draw_detections(self, frame, detected_texts):
        """Dibujar las detecciones de texto en el frame"""
        annotated_frame = frame.copy()

        # Estadísticas por tipo
        type_counts = {"letters": 0, "numbers": 0, "mixed": 0, "symbols": 0}

        for i, detection in enumerate(detected_texts):
            x1, y1, x2, y2 = detection["bbox"]
            text = detection["text"]
            confidence = detection["confidence"]
            text_type = detection["type"]

            # Contar tipos
            type_counts[text_type] += 1

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

        # Estadísticas por tipo
        y_offset = 60
        for text_type, count in type_counts.items():
            if count > 0:
                color = self.colors[text_type]
                cv2.putText(
                    annotated_frame,
                    f"{text_type.title()}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                y_offset += 20

        # Información del modelo
        model_info = "YOLOv8 + EasyOCR - Detección de Caracteres"
        cv2.putText(
            annotated_frame,
            model_info,
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return annotated_frame

    def get_text_statistics(self, detected_texts):
        """Obtener estadísticas del texto detectado"""
        if not detected_texts:
            return {}

        # Contar caracteres por tipo
        char_counts = {"letters": 0, "numbers": 0, "symbols": 0}
        text_by_type = {"letters": [], "numbers": [], "mixed": [], "symbols": []}

        for detection in detected_texts:
            text = detection["text"]
            text_type = detection["type"]
            text_by_type[text_type].append(text)

            # Contar caracteres individuales
            for char in text:
                if char.isalpha():
                    char_counts["letters"] += 1
                elif char.isdigit():
                    char_counts["numbers"] += 1
                elif not char.isspace():
                    char_counts["symbols"] += 1

        stats = {
            "total_detections": len(detected_texts),
            "character_counts": char_counts,
            "text_by_type": text_by_type,
            "avg_confidence": np.mean([d["confidence"] for d in detected_texts]),
        }

        return stats


def main():
    """Función principal para ejecutar la detección de caracteres en tiempo real"""
    print("=== DETECCIÓN DE CARACTERES/TEXTO EN TIEMPO REAL ===")
    print("Presiona 'q' para salir")
    print("Presiona 's' para mostrar estadísticas detalladas")
    print("Presiona 'r' para resetear estadísticas")
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
    window_name = "Detección de Caracteres/Texto - YOLOv8 + OCR"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    show_detailed_stats = False
    paused = False
    all_detected_texts = []

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ No se pudo leer el frame de la cámara.")
                break

            frame_count += 1

            # Detectar texto/caracteres
            detected_texts = detector.detect_characters(frame)
            all_detected_texts.extend(detected_texts)

            # Dibujar detecciones
            annotated_frame = detector.draw_detections(frame, detected_texts)

            # Mostrar estadísticas detalladas si está activado
            if show_detailed_stats and detected_texts:
                stats = detector.get_text_statistics(detected_texts)

                # Mostrar textos únicos detectados
                y_pos = 100
                for text_type, texts in stats["text_by_type"].items():
                    if texts:
                        unique_texts = list(set(texts))[:3]  # Máximo 3 únicos
                        color = detector.colors[text_type]
                        texts_str = ", ".join(unique_texts)
                        cv2.putText(
                            annotated_frame,
                            f"{text_type}: {texts_str}",
                            (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )
                        y_pos += 15
        else:
            # Si está pausado, mantener el último frame
            pass

        # Mostrar frame
        cv2.imshow(window_name, annotated_frame)

        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            show_detailed_stats = not show_detailed_stats
            print(
                f"Estadísticas detalladas: {'activadas' if show_detailed_stats else 'desactivadas'}"
            )
        elif key == ord("r"):
            all_detected_texts = []
            print("Estadísticas reseteadas")
        elif key == ord("p"):
            paused = not paused
            print(f"{'Pausado' if paused else 'Reanudado'}")

    # Mostrar estadísticas finales
    if all_detected_texts:
        final_stats = detector.get_text_statistics(all_detected_texts)
        print("\n=== ESTADÍSTICAS FINALES ===")
        print(f"Total detecciones: {final_stats['total_detections']}")
        print(f"Caracteres - Letras: {final_stats['character_counts']['letters']}")
        print(f"Caracteres - Números: {final_stats['character_counts']['numbers']}")
        print(f"Caracteres - Símbolos: {final_stats['character_counts']['symbols']}")
        print(f"Confianza promedio: {final_stats['avg_confidence']:.2f}")

    # Limpieza
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Programa terminado. Frames procesados: {frame_count}")


if __name__ == "__main__":
    main()
