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
import time
import csv
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
from ultralytics import YOLO


# ===============================
# DESCARGA DE ARCHIVOS
# ===============================
def download_file(url, filename):
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
    configs = {
        "4": {
            "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights",
            "model_file": None,
        },
        "5": {"cfg": None, "weights": None, "model_file": "yolov5s.pt"},
        "6": {"cfg": None, "weights": None, "model_file": "yolov6s.pt"},
        "7": {"cfg": None, "weights": None, "model_file": "yolov7.pt"},
        "8": {"cfg": None, "weights": None, "model_file": "yolov8n.pt"},
    }
    return configs.get(version, configs["8"])


def download_yolo_files(version="8"):
    download_file(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "coco.names",
    )
    config = get_yolo_files_config(version)
    if version == "4":
        download_file(config["cfg"], "yolov4.cfg")
        download_file(config["weights"], "yolov4.weights")
    else:
        print(f"✓ YOLOv{version} se descargará automáticamente")
    return config


# ===============================
# CLASE BASE
# ===============================
class BaseYOLODetector(ABC):
    def __init__(self, version, confidence=0.5, class_names=None):
        self.version = version
        self.confidence = confidence
        self.class_names = class_names or []

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def detect(self, frame):
        pass

    @abstractmethod
    def draw_detections(self, frame, detections):
        pass

    def get_model_info(self):
        return f"YOLOv{self.version}"


# ===============================
# YOLOv4 - OpenCV DNN
# ===============================
class YOLOv4Detector(BaseYOLODetector):
    def __init__(self, version, confidence=0.5, class_names=None):
        super().__init__(version, confidence, class_names)
        self.net = None
        self.output_layers = None
        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def load_model(self):
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    cx, cy, bw, bh = (
                        int(det[0] * w),
                        int(det[1] * h),
                        int(det[2] * w),
                        int(det[3] * h),
                    )
                    x, y = int(cx - bw / 2), int(cy - bh / 2)
                    boxes.append([x, y, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, 0.4)
        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(
                    {
                        "box": boxes[i],
                        "confidence": confidences[i],
                        "class_id": class_ids[i],
                        "label": self.class_names[class_ids[i]],
                    }
                )
        return detections

    def draw_detections(self, frame, detections):
        for det in detections:
            x, y, w, h = det["box"]
            color = self.colors[det["class_id"] % 80]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{det['label']} {det['confidence']:.2f}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return frame


# ===============================
# YOLOv5–YOLOv8 - Ultralytics
# ===============================
class UltralyticsDetector(BaseYOLODetector):
    def __init__(self, version, confidence=0.5, class_names=None, model_file=None):
        super().__init__(version, confidence, class_names)
        self.model_file = model_file
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_file)
        self.model.to("cpu")

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cid = int(box.cls[0])
                detections.append(
                    {
                        "confidence": float(box.conf[0]),
                        "class_id": cid,
                        "label": self.class_names[cid],
                    }
                )
        return detections, results

    def draw_detections(self, frame, detection_data):
        detections, results = detection_data
        return results[0].plot()


# ===============================
# FACTORY
# ===============================
def create_detector(version, confidence, class_names):
    config = get_yolo_files_config(version)
    if version == "4":
        detector = YOLOv4Detector(version, confidence, class_names)
    else:
        detector = UltralyticsDetector(
            version, confidence, class_names, config["model_file"]
        )
    detector.load_model()
    return detector


# ===============================
# MAIN
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", default="8", choices=["4", "5", "6", "7", "8"])
    parser.add_argument("-c", "--confidence", type=float, default=0.5)
    parser.add_argument("-r", "--resolution", default="640x480")
    args = parser.parse_args()

    download_yolo_files(args.version)

    with open("coco.names") as f:
        class_names = [line.strip() for line in f.readlines()]

    width, height = map(int, args.resolution.split("x"))
    detector = create_detector(args.version, args.confidence, class_names)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    csv_file = open("detections_log.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "frame", "class", "confidence"])

    class_counter = defaultdict(int)
    prev_time = time.time()
    frame_count = 0

    alpha = 1.0  # contraste

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

        frame_count += 1
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        detections = detector.detect(frame)
        detections_list = detections[0] if isinstance(detections, tuple) else detections

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for det in detections_list:
            class_counter[det["label"]] += 1
            csv_writer.writerow(
                [timestamp, frame_count, det["label"], det["confidence"]]
            )

        annotated_frame = detector.draw_detections(frame.copy(), detections)

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Contraste: {alpha:.1f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 0),
            2,
        )

        y = 80
        for label, count in list(class_counter.items())[:5]:
            cv2.putText(
                annotated_frame,
                f"{label}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),  # 🟠 NARANJA
                2,
            )
            y += 20

        cv2.imshow("YOLO Detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("+"):
            alpha = min(alpha + 0.1, 3.0)
        elif key == ord("-"):
            alpha = max(alpha - 0.1, 0.3)
        elif key == ord("q"):
            break

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Programa finalizado")


if __name__ == "__main__":
    main()
