# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "ultralytics",
#     "requests",
#     "argparse",
# ]
# ///
import cv2
import numpy as np
import os
import requests
import argparse
from abc import ABC, abstractmethod
from ultralytics import YOLO


def download_file(url, filename):
    """Descargar archivo si no existe"""
    if not os.path.exists(filename):
        print(f"Descargando {filename}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✓ {filename} descargado")
        except Exception as e:
            print(f"❌ Error descargando {filename}: {e}")
    else:
        print(f"✓ {filename} ya existe")


def get_yolo_files_config(version):
    """Obtener configuración de archivos según la versión de YOLO"""
    configs = {
        "4": {
            "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            "model_file": None,
        },
        "5": {
            "cfg": None,
            "weights": None,
            "model_file": "yolov5s.pt",
        },
        "6": {
            "cfg": None,
            "weights": None,
            "model_file": "yolov6s.pt",
        },
        "7": {
            "cfg": None,
            "weights": None,
            "model_file": "yolov7.pt",
        },
        "8": {
            "cfg": None,
            "weights": None,
            "model_file": "yolov8n.pt",
        },
    }
    return configs.get(version, configs["8"])


def download_yolo_files(version="8"):
    """Descargar archivos necesarios según la versión de YOLO"""
    print(f"Configurando YOLOv{version}...")

    # Siempre descargar coco.names
    download_file(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "coco.names",
    )

    config = get_yolo_files_config(version)

    if version == "4":
        # YOLOv4 necesita cfg y weights
        download_file(config["cfg"], "yolov4.cfg")
        download_file(config["weights"], "yolov4.weights")
    else:
        # YOLOv5-8 usan ultralytics, se descargan automáticamente
        print(f"✓ YOLOv{version} se descargará automáticamente la primera vez")

    print("Todos los archivos están listos!\n")
    return config


class BaseYOLODetector(ABC):
    """Clase base abstracta para detectores YOLO"""

    def __init__(self, version, confidence=0.5, class_names=None):
        self.version = version
        self.confidence = confidence
        self.class_names = class_names or []

    @abstractmethod
    def load_model(self):
        """Cargar el modelo YOLO"""
        pass

    @abstractmethod
    def detect(self, frame):
        """Detectar objetos en un frame"""
        pass

    @abstractmethod
    def draw_detections(self, frame, detections):
        """Dibujar detecciones en el frame"""
        pass

    def get_model_info(self):
        """Obtener información del modelo"""
        return f"YOLOv{self.version}"


class YOLOv4Detector(BaseYOLODetector):
    """Detector para YOLOv4 usando OpenCV DNN"""

    def __init__(self, version, confidence=0.5, class_names=None):
        super().__init__(version, confidence, class_names)
        self.net = None
        self.output_layers = None
        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def load_model(self):
        """Cargar modelo YOLOv4 usando OpenCV DNN"""
        print("Cargando YOLOv4 con OpenCV DNN...")
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        print(f"✓ Modelo YOLOv4 cargado con {len(layer_names)} capas")

    def detect(self, frame):
        """Detectar objetos usando YOLOv4 con OpenCV DNN"""
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, 0.4)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                label = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Class {class_id}"
                )
                detections.append(
                    {
                        "box": [x, y, w, h],
                        "confidence": confidence,
                        "class_id": class_id,
                        "label": label,
                    }
                )

        return detections

    def draw_detections(self, frame, detections):
        """Dibujar detecciones para YOLOv4"""
        for detection in detections:
            x, y, w, h = detection["box"]
            confidence = detection["confidence"]
            label = detection["label"]
            class_id = detection["class_id"]

            color = self.colors[class_id % 80]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(
                frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Información adicional
        cv2.putText(
            frame,
            f"YOLOv4 (OpenCV) | Detecciones: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if detections:
            detected_objects = [d["label"] for d in detections[:3]]
            classes_text = ", ".join(detected_objects)
            cv2.putText(
                frame,
                f"Objetos: {classes_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        return frame

    def get_model_info(self):
        return f"YOLOv{self.version} (OpenCV DNN)"


class UltralyticsDetector(BaseYOLODetector):
    """Detector para YOLOv5-v8 usando Ultralytics"""

    def __init__(self, version, confidence=0.5, class_names=None, model_file=None):
        super().__init__(version, confidence, class_names)
        self.model = None
        self.model_file = model_file

    def load_model(self):
        """Cargar modelo usando Ultralytics"""
        print(f"Cargando YOLOv{self.version} con Ultralytics...")
        self.model = YOLO(self.model_file)
        self.model.to("cpu")
        print(f"✓ Modelo YOLOv{self.version} cargado")

    def detect(self, frame):
        """Detectar objetos usando Ultralytics"""
        results = self.model(frame, verbose=False, conf=self.confidence)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"Class {class_id}"
                )
                detections.append(
                    {"confidence": confidence, "class_id": class_id, "label": label}
                )

        return detections, results

    def draw_detections(self, frame, detection_data):
        """Dibujar detecciones para Ultralytics"""
        detections, results = detection_data
        annotated_frame = results[0].plot()

        # Información adicional
        cv2.putText(
            annotated_frame,
            f"YOLOv{self.version} | Detecciones: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if detections:
            detected_objects = [d["label"] for d in detections[:3]]
            classes_text = ", ".join(detected_objects)
            cv2.putText(
                annotated_frame,
                f"Objetos: {classes_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        return annotated_frame

    def get_model_info(self):
        return f"YOLOv{self.version} (Ultralytics)"


def create_detector(version, confidence, class_names):
    """Factory function para crear el detector apropiado"""
    config = get_yolo_files_config(version)

    if version == "4":
        detector = YOLOv4Detector(version, confidence, class_names)
    else:
        detector = UltralyticsDetector(
            version, confidence, class_names, config["model_file"]
        )

    detector.load_model()
    return detector


def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Detección de objetos en tiempo real con YOLO"
    )
    parser.add_argument(
        "--version",
        "-v",
        default="8",
        choices=["4", "5", "6", "7", "8"],
        help="Versión de YOLO a usar (4-8). Por defecto: 8",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.5,
        help="Umbral de confianza (0.0-1.0). Por defecto: 0.5",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        default="640x480",
        help="Resolución de cámara (ej: 640x480, 1280x720). Por defecto: 640x480",
    )

    args = parser.parse_args()

    print(f"=== DETECCIÓN EN TIEMPO REAL CON YOLOv{args.version} ===")
    print("Presiona 'q' para salir.\n")

    # Descargar archivos según la versión
    download_yolo_files(args.version)

    # Cargar nombres de clases COCO
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Cargadas {len(class_names)} clases de COCO")

    # Configurar resolución
    width, height = map(int, args.resolution.split("x"))

    # Crear detector usando factory pattern
    detector = create_detector(args.version, args.confidence, class_names)

    # Configurar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(f"✓ Cámara configurada a {width}x{height}")
    print(f"✓ Usando {detector.get_model_info()}")
    print(f"✓ Umbral de confianza: {args.confidence}")

    # Crear ventana
    window_name = f"YOLOv{args.version} - Detección de Objetos"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame de la cámara.")
            break

        frame_count += 1

        # Realizar detección usando la clase apropiada
        detections = detector.detect(frame)

        # Dibujar detecciones usando la clase apropiada
        annotated_frame = detector.draw_detections(frame.copy(), detections)

        # Mostrar frame
        cv2.imshow(window_name, annotated_frame)

        # Verificar tecla de salida
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Programa terminado. Frames procesados: {frame_count}")


if __name__ == "__main__":
    main()
