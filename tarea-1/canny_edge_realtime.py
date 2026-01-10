# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "opencv-python",
# ]
# ///
import cv2
import numpy as np

# Explicación de los thresholds:
# threshold1: Valor mínimo. Los gradientes de píxeles por debajo de este valor se descartan como no bordes.
# threshold2: Valor máximo. Los gradientes de píxeles por encima de este valor se consideran bordes definitivos.
# Los valores entre threshold1 y threshold2 se consideran bordes solo si están conectados a un borde fuerte.

def main():
    threshold1 = 100  # Umbral mínimo
    threshold2 = 200  # Umbral máximo
    print("Presiona 'q' para salir. Usa las teclas 'w/s' para aumentar/disminuir el umbral 1 y 'e/d' para el umbral 2.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)

        # Crear imagen de bordes en blanco sobre fondo negro
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, edges_rgb))

        cv2.putText(combined, f"Umbral 1: {threshold1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, f"Umbral 2: {threshold2}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('1. Original | 2. Bordes', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            threshold1 = min(255, threshold1 + 1)
        elif key == ord('s'):
            threshold1 = max(0, threshold1 - 1)
        elif key == ord('e'):
            threshold2 = min(255, threshold2 + 1)
        elif key == ord('d'):
            threshold2 = max(0, threshold2 - 1)
        # Evitar que threshold1 >= threshold2
        if threshold1 >= threshold2:
            threshold1 = threshold2 - 1 if threshold2 > 0 else 0

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
