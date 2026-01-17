# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
#     "ultralytics",
# ]
# ///
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    print("Presiona 'q' para salir.")
    print("Detección de objetos en tiempo real con YOLO Tiny")

    # Usar YOLOv8 nano (tiny) modelo pre-entrenado
    # Esto descargará el modelo automáticamente la primera vez
    model = YOLO("yolov8n.pt")  # nano es el más pequeño y rápido
    model.to('cpu')  # Forzar CPU debido a problemas de compatibilidad CUDA
    
    # Configurar la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return
    
    # Configurar resolución para mejor rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Crear ventana una sola vez
    window_name = 'YOLO - Detección de Objetos en Tiempo Real'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Realizar detección
        results = model(frame, verbose=False)
        
        # Dibujar detecciones en el frame
        annotated_frame = results[0].plot()
        
        # Mostrar FPS y número de detecciones
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Detecciones: {detections}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Actualizar el frame en la ventana existente
        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
